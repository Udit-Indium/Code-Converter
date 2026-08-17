import ast
import base64
import json
import os
import pathlib
import re
import time
import uuid
import requests
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from dotenv import load_dotenv
load_dotenv()


OUTPUTS_DIR = pathlib.Path(__file__).parents[3] / "outputs"
_FILE_HEADER = '"""Auto-assembled PySpark conversion (built incrementally, one batch\nof functions per loop iteration)."""\n'
_MAX_LOG_CHARS = 3000

_SKILL_DIR = pathlib.Path(__file__).parent / "skills" / "py2snow-skill"


def _tail(text: "str | None", limit: int = _MAX_LOG_CHARS) -> str:
    """Keep only the last `limit` chars of tool output so verbose PySpark logs
    don't accumulate in the LLM context window."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return "…[truncated earlier output]…\n" + text[-limit:]


def _error_summary(stderr: "str | None", file_path: str, limit: int = 400) -> str:
    """Extract an actionable one-liner from a Python traceback: which function in
    the converted file failed (line + name) and the exception message. Lets the
    converter fix the exact function instead of scanning raw Spark log spam."""
    if not stderr:
        return ""
    fname = os.path.basename(file_path or "")
    lines = stderr.splitlines()
    exc = ""
    for l in reversed(lines):
        s = l.strip()
        if s and ("Error" in s or "Exception" in s) and ":" in s \
                and not s.startswith(("WARN", "File ")):
            exc = s
            break
    where = ""
    for l in lines:
        s = l.strip()
        if fname and fname in s and ", in " in s:  # File ".../x.py", line N, in func
            where = " " + s.split(fname, 1)[1].strip().lstrip('",').strip()
    summary = (f"Crashed at{where}. {exc}").strip()
    return summary[:limit]


def _as_dict(value):
    """Tolerate an inventory that is already a dict, or is raw JSON text."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}

AST_INVENTORY = OUTPUTS_DIR / "ast_inventory.json"

# Cap on a single constant's string value in read_source_index_tool, so one
# embedded SQL blob cannot crowd out every other constant in the response.
MAX_CONSTANT_VALUE_CHARS = 500


def _inventory() -> dict:
    """Load the parsed AST inventory from disk.

    The parser deliberately does NOT put this in state: for a script of a few
    hundred functions it is tens of kilobytes of JSON, and state is echoed into
    every prompt of every agent in the session.

    Returns an empty dict when the parser has not run or the file is
    unreadable — callers already treat an empty inventory as "nothing to do",
    so raising would turn an ordering problem into a crash.
    """
    try:
        return _as_dict(AST_INVENTORY.read_text(encoding="utf-8"))
    except OSError:
        return {}


def _source_function_names() -> list[str]:
    """Every function name in the parsed SOURCE script (order preserved)."""
    apc = _inventory()
    return [
        f.get("name")
        for f in (apc.get("functions") or [])
        if isinstance(f, dict) and f.get("name")
    ]


def _canonical_output_path(state) -> pathlib.Path:
    """The ONE stable file every batch is appended to.

    Derived once from the source script name and cached in state so retries and
    the downstream fixer agents all target the same file (prevents the
    `_spark.py` vs `_spark_complete.py` divergence).
    """
    existing = state.get("converted_pyspark_file_path")
    if existing:
        return pathlib.Path(existing)
    stem = "converted"
    apc = _inventory()
    script_path = (apc.get("metadata") or {}).get("script_path")
    if script_path:
        stem = pathlib.Path(script_path).stem
    # Must be importable: the parity suite and the semantic runner both do
    # `from <stem>_spark import ...`.
    stem = re.sub(r"[^0-9a-zA-Z_]+", "_", stem).strip("_").lower() or "converted"
    if stem[0].isdigit():
        stem = f"s_{stem}"
    p = OUTPUTS_DIR / f"{stem}_spark.py"
    state["converted_pyspark_file_path"] = str(p)
    return p


def _module_const_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _pandas_violations(src: str) -> list[str]:
    """Find pandas idioms in submitted code that must be native Spark instead.

    This is the rule the conventions repeated hardest, and repetition is an
    expensive way to enforce anything — it cost ~10k tokens of prompt on every
    turn and the model could still ignore it. Checking here is deterministic and
    unskippable, so the prompt only has to STATE the rule once.

    Deliberately high-confidence only. numpy is NOT flagged: data-generation
    functions are required to build rows in plain Python (with seeded numpy /
    random) and hand them to spark.createDataFrame — banning it would reject
    correct conversions.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []          # the caller reports the syntax error itself

    found: list[str] = []
    for node in ast.walk(tree):
        # import pandas / from pandas import ...
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "pandas":
                    found.append(f"line {node.lineno}: `import pandas` — use the Spark DataFrame API")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "pandas":
                found.append(f"line {node.lineno}: `from pandas import …` — use the Spark DataFrame API")

        elif isinstance(node, ast.Attribute):
            # pd.<anything>
            if isinstance(node.value, ast.Name) and node.value.id in ("pd", "pandas"):
                found.append(f"line {node.lineno}: `{node.value.id}.{node.attr}` — pandas call, convert to Spark")
            # .iloc / .loc positional indexing
            elif node.attr in ("iloc", "loc"):
                found.append(f"line {node.lineno}: `.{node.attr}` — no positional indexing in Spark; use filter/select")

        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if name == "merge":
                found.append(f"line {node.lineno}: `.merge(...)` — use `.join(other, on=…, how=…)`")
            elif name == "toPandas":
                found.append(f"line {node.lineno}: `.toPandas()` — collapses the frame to the driver")
            elif name == "rename" and any(k.arg == "columns" for k in node.keywords):
                found.append(f"line {node.lineno}: `.rename(columns=…)` — use `.withColumnRenamed(old, new)`")

    # de-dupe, keep order
    return list(dict.fromkeys(found))


def _merge_snippet(existing_src: str, snippet: str) -> tuple[str, list[str]]:
    """Deterministically merge a batch of new code into the existing file.

    Keeps a single import block + constants + function/class defs, de-duping by
    name so a batch never clobbers or duplicates already-converted functions.
    The file is rebuilt in Python (not by the LLM), so it never truncates no
    matter how many functions accumulate. Returns (new_source, added_names).
    """
    return _assemble(existing_src, snippet, replace=False)


def _raw_segments(src: str):
    """Split module source into top-level segments, keeping the ORIGINAL text of
    each (comments and formatting intact). ast is used only to locate line spans.

    Yields tuples (kind, key, raw_text) where kind is
    'import' | 'const' | 'def' | 'other'. Any `#` comment lines sitting directly
    above a segment are attached to it.
    """
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    prev_end = 0 
    for node in tree.body:
        node_start = node.lineno
        if getattr(node, "decorator_list", None):
            node_start = min(d.lineno for d in node.decorator_list)
        top = node_start - 1
        while top - 1 >= prev_end and lines[top - 1].strip().startswith("#"):
            top -= 1
        end = node.end_lineno
        raw = "".join(lines[top:end]).rstrip("\n")
        prev_end = end
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            continue

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield "import", ast.unparse(node), raw, frozenset()
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield "def", node.name, raw, frozenset()
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            key = tuple(sorted(_module_const_names(
                ast.Module(body=[node], type_ignores=[]))))
            rhs = node.value
            refs = frozenset(
                n.id for n in ast.walk(rhs) if isinstance(n, ast.Name)
            ) if rhs is not None else frozenset()
            yield "const", (key or ("<expr>",)), raw, refs
        else:
            yield "other", None, raw, frozenset()


def _assemble(existing_src: str, snippet: str, replace: bool) -> tuple[str, list[str]]:
    """Rebuild the file from existing content + a snippet, deterministically,
    PRESERVING the original source text of each piece (comments included).

    Guarantees a well-formed library module:
      * imports are hoisted (deduped) to the top,
      * constants and defs keep their ENCOUNTER ORDER (constants are NEVER
        reordered above the functions — that was what produced
        `NameError: <fn> not defined` when a stray top-level call got hoisted),
      * stray top-level executable code and `__main__` guards are DROPPED (a
        converted function library must not run anything on import), and
      * an assignment whose right-hand side CALLS a locally-defined function
        (e.g. `order_df = make_orders_df()`) is DROPPED as demo/scratch code.

    For functions/classes:
      * replace=False (append/convert): a name already present is KEPT as-is.
      * replace=True (fix): a name already present is OVERWRITTEN in place.
    Returns (new_source, changed_names).
    """
    import_order: list[str] = []
    imports: dict[str, str] = {}
    body_order: list[tuple] = []         
    body: dict[tuple, str] = {}
    refs: dict[tuple, frozenset] = {}   
    def_names: set = set()
    changed: list[str] = []

    def _ingest(src: str, is_snippet: bool):
        if not src.strip():
            return
        for kind, key, raw, node_refs in _raw_segments(src):
            if kind == "import":
                if key not in imports:
                    import_order.append(key)
                    imports[key] = raw
            elif kind == "def":
                def_names.add(key)
                bk = ("def", key)
                if bk not in body:
                    body_order.append(bk)
                    body[bk] = raw
                    if is_snippet:
                        changed.append(key)
                elif is_snippet and replace:
                    body[bk] = raw           # overwrite in place, keep position
                    changed.append(key)
            elif kind == "const":
                bk = ("const", key)
                if bk not in body:
                    body_order.append(bk)
                    body[bk] = raw
                    refs[bk] = node_refs
                elif is_snippet and replace:
                    body[bk] = raw
                    refs[bk] = node_refs
            # 'other' is dropped

    _ingest(existing_src, False)   # existing content first (defines order)
    _ingest(snippet, True)         # then the snippet

    # Drop assignments that call a locally-defined function (stray demo code).
    kept = [
        bk for bk in body_order
        if not (bk[0] == "const" and (refs.get(bk, frozenset()) & def_names))
    ]

    parts = [_FILE_HEADER.rstrip("\n")]
    if import_order:
        parts.append("\n".join(imports[k] for k in import_order))
    if kept:
        parts.append("\n\n\n".join(body[bk] for bk in kept))
    return "\n\n".join(parts).rstrip() + "\n", changed


PROGRESS_FILE = OUTPUTS_DIR / "migration_progress.json"


def _write_progress(state, converted: set[str]) -> dict:
    """Record migration progress to outputs/migration_progress.json.

    Written after every batch so the run's state survives outside the LLM
    context — the agent reads it back with read_migration_progress_tool instead
    of us re-injecting the whole picture into its prompt each turn.
    """
    source = _source_function_names()
    remaining = [n for n in source if n not in converted]
    progress = {
        "source_function_count": len(source),
        "converted_count": len(source) - len(remaining),
        "remaining_count": len(remaining),
        "percent_complete": round(
            100.0 * (len(source) - len(remaining)) / len(source), 1
        ) if source else 0.0,
        "converted": [n for n in source if n in converted],
        "remaining": remaining,
        "extra_in_output": sorted(converted - set(source)),
        "output_file": state.get("converted_pyspark_file_path"),
    }
    try:
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    except OSError:
        pass
    return progress

SOURCE_SCRIPT = OUTPUTS_DIR / "source_script.py"


def _source_text() -> str:
    """The source script, read from the parser's canonical output path.

    Read from a fixed path rather than from a path published in state. The
    parser deliberately puts neither the script nor its location in state, so
    the previous `state.get("source_script_path")` returned None and every tool
    built on this silently saw an empty source: no function index, no bodies to
    convert, and no error explaining it.
    """
    try:
        return SOURCE_SCRIPT.read_text(encoding="utf-8")
    except OSError:
        return ""


def read_source_index_tool(context: ToolContext) -> dict:
    """Return a compact source overview without function bodies.

    Normally the converter does not need this tool because the next batch is
    already injected by _next_batch_source(). Use it only when metadata is
    genuinely required.
    """
    src = _source_text()
    if not src.strip():
        return {"available": False, "count": 0, "functions": [],
                "error": "source script not found"}
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {"available": False, "count": 0, "functions": [],
                "error": f"source has a syntax error: {exc}"}

    functions = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        entry = {"name": node.name, "kind": type(node).__name__}
        if not isinstance(node, ast.ClassDef):
            entry["parameters"] = [arg.arg for arg in node.args.args]
        functions.append(entry)

    return {
        "available": True,
        "count": len(functions),
        "functions": functions,
        # Values, not just names. The case-fact checker reports a constant
        # mismatch as a bare NAME, so if this tool also returned only names the
        # converter had no way at all to learn the value it was supposed to
        # write — it would burn its turn hunting for the raw source through the
        # skill tools and then guess. The parser already stores the literal
        # values in the inventory; hand them over.
        "constants": _source_constants(),
    }


def _source_constants() -> dict:
    """Module-level constant name -> literal value, from the AST inventory.

    Long values are truncated with a marker rather than dropped: a constant the
    converter can see is wrong is far more useful than one it cannot see at all.
    """
    consts = _inventory().get("constants") or {}
    if not isinstance(consts, dict):
        return {}
    out = {}
    for name in sorted(consts):
        value = consts[name]
        if isinstance(value, str) and len(value) > MAX_CONSTANT_VALUE_CHARS:
            value = value[:MAX_CONSTANT_VALUE_CHARS] + "...<truncated>"
        out[name] = value
    return out


def read_migration_progress_tool(context: ToolContext) -> dict:
    """How much of the migration is done: converted vs remaining, by name.

    Read from outputs/migration_progress.json, refreshed after every batch. Use
    it to confirm what is left rather than assuming.
    """
    try:
        progress = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        # Keep the tool response compact; the full lists remain on disk.
        return {
            "source_function_count": progress.get("source_function_count", 0),
            "converted_count": progress.get("converted_count", 0),
            "remaining_count": progress.get("remaining_count", 0),
            "percent_complete": progress.get("percent_complete", 0.0),
            "remaining_sample": (progress.get("remaining") or [])[:20],
            "extra_in_output_count": len(progress.get("extra_in_output") or []),
            "output_file": progress.get("output_file"),
        }
    except (OSError, ValueError):
        source = _source_function_names()
        return {"source_function_count": len(source), "converted_count": 0,
                "remaining_count": len(source), "percent_complete": 0.0,
                "remaining_sample": source[:20], "extra_in_output_count": 0,
                "output_file": context.state.get("converted_pyspark_file_path")}


def add_converted_functions_tool(context: ToolContext, functions_code: str) -> dict:
    """Append a BATCH of newly-converted PySpark code to the single output file.

    Pass ONLY the functions/classes you converted this turn (plus any imports or
    module-level constants they need) — never re-send functions already in the
    file. The file is reassembled deterministically in Python, so functions
    accumulate across loop iterations without you ever having to reproduce the
    whole file (which is what causes truncation / `# continue similarly` stubs).
    """
    p = _canonical_output_path(context.state)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = p.read_text(encoding="utf-8") if p.exists() else ""
    try:
        merged, added = _merge_snippet(existing, functions_code)
    except SyntaxError as exc:
        return {
            "status": "error",
            "error": f"The submitted functions_code has a syntax error: {exc}. "
                     "Fix it and resubmit only that batch.",
        }
    p.write_text(merged, encoding="utf-8")

    # Refresh the progress file, and report the SHORT form back — counts and
    # what is left, not the full list of everything already done.
    total = {
        n.name
        for n in ast.parse(merged).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    progress = _write_progress(context.state, total)
    result = {
        "status": "success",
        "saved_file_path": str(p),
        "functions_added_this_batch": added,
        "converted_count": progress["converted_count"],
        "remaining_count": progress["remaining_count"],
        "percent_complete": progress["percent_complete"],
        "remaining_sample": progress["remaining"][:20],
        "remaining_truncated": len(progress["remaining"]) > 20,
    }

    # Deterministic check of the rule the conventions repeat hardest. Reported
    # rather than rejected: `.toPandas()` on a SMALL aggregate is required by
    # the visualisation conventions, so a hard block would refuse correct
    # plotting conversions. The batch is already saved; this tells the model
    # exactly what to fix with replace_functions_tool.
    violations = _pandas_violations(functions_code)
    if violations:
        result["pandas_violations"] = violations
        result["action_required"] = (
            "This batch uses pandas idioms that must be native Spark. Fix them "
            "with replace_functions_tool before moving on. If a `.toPandas()` is "
            "a deliberate small-aggregate collect for plotting, say so and keep it."
        )
    return result


def replace_functions_tool(context: ToolContext, functions_code: str) -> dict:
    """Surgically REPLACE specific functions in the converted file, in place.

    Used by the fixer agents. Submit ONLY the corrected version(s) of the
    function(s) you are fixing (plus any new imports/constants they need). Each
    function in `functions_code` overwrites the same-named function already in
    the file; every OTHER function is left byte-for-byte untouched. Brand-new
    names are appended. This means you NEVER reproduce the whole file — so a big
    file can't get truncated into half-converted code. Fix in small batches.
    """
    p = _canonical_output_path(context.state)
    if not p.exists():
        return {
            "status": "error",
            "error": "No converted file exists yet — nothing to fix.",
        }
    existing = p.read_text(encoding="utf-8")
    try:
        merged, changed = _assemble(existing, functions_code, replace=True)
    except SyntaxError as exc:
        return {
            "status": "error",
            "error": f"The submitted functions_code has a syntax error: {exc}. "
                     "Fix it and resubmit only that batch.",
        }
    p.write_text(merged, encoding="utf-8")
    total = {
        n.name
        for n in ast.parse(merged).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return {
        "status": "success",
        "saved_file_path": str(p),
        "functions_replaced_or_added": changed,
        "total_function_count": len(total),
    }


def read_functions_tool(context: ToolContext, function_names: list[str]) -> dict:
    """Return the current source of ONLY the named functions from the converted
    file (so a fixer can inspect just the ones it needs without pulling the whole
    file into context). Unknown names are reported under `not_found`."""
    p = _canonical_output_path(context.state)
    if not p.exists():
        return {"exists": False, "functions": {}, "not_found": list(function_names or [])}
    tree = ast.parse(p.read_text(encoding="utf-8"))
    by_name = {
        n.name: ast.unparse(n)
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    requested = list(function_names or [])
    found = {name: by_name[name] for name in requested if name in by_name}
    not_found = [name for name in requested if name not in by_name]
    return {"exists": True, "functions": found, "not_found": not_found}


def read_source_functions_tool(context: ToolContext, function_names: list[str]) -> dict:
    """Return the ORIGINAL Python source of the named functions (ground truth).

    Use this before fixing/converting a function so you match the source's real
    behaviour instead of guessing. Read from outputs/source_script.py — pull only
    the handful you are working on this turn, never the whole file.
    Unknown names are listed under `not_found`."""
    src = _source_text()
    if not src.strip():
        return {"available": False, "functions": {}, "not_found": list(function_names or [])}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {"available": False, "functions": {}, "not_found": list(function_names or [])}
    by_name = {
        n.name: (ast.get_source_segment(src, n) or "")
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    requested = list(function_names or [])
    found = {name: by_name[name] for name in requested if name in by_name}
    not_found = [name for name in requested if name not in by_name]
    return {"available": True, "functions": found, "not_found": not_found}


def read_converted_file_tool(context: ToolContext) -> dict:
    """List what is in the converted file: function names, and how big it is.

    Deliberately does NOT return the code. This answers "which functions exist?",
    which is the question it is almost always asked; pulling the whole file to
    answer it put thousands of tokens into context that then rode along in every
    later request. When you need a body, call **read_functions_tool** with the
    names you got from here."""
    p = _canonical_output_path(context.state)
    if not p.exists():
        return {"exists": False, "function_names": [], "line_count": 0}
    src = p.read_text(encoding="utf-8")
    names = sorted(
        n.name
        for n in ast.parse(src).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    return {"exists": True, "function_names": names,
            "line_count": len(src.splitlines())}



BATCH_SIZE = 4


def _next_batch_source(state) -> str:
    """Source of the next BATCH_SIZE functions still to convert.

    Injected directly rather than fetched with a tool. A tool call costs a whole
    extra model round-trip, and every round-trip re-sends the ~10k-token
    conventions block — so paying ~2k of prompt here saves ~12k of round-trip.
    The model gets exactly the same bodies either way.
    """
    # Derive the work-list from disk instead of storing the full list in ADK state.
    # ADK may carry state into every model turn, so keeping hundreds of function
    # names in state can become a large repeated prompt payload.
    source_names = _source_function_names()
    output_path = _canonical_output_path(state)
    converted_names: set[str] = set()
    if output_path.exists():
        try:
            tree = ast.parse(output_path.read_text(encoding="utf-8"))
            converted_names = {
                n.name for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
        except (OSError, SyntaxError):
            converted_names = set()
    missing = [name for name in source_names if name not in converted_names]
    if not missing:
        return "(nothing left to convert)"
    batch = missing[:BATCH_SIZE]
    src = _source_text()
    if not src.strip():
        return "(source unavailable — use read_source_functions_tool)"
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "(source does not parse — use read_source_functions_tool)"
    by_name = {
        n.name: (ast.get_source_segment(src, n) or "")
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    parts = [by_name[n] for n in batch if by_name.get(n)]
    missing_bodies = [n for n in batch if not by_name.get(n)]
    out = "\n\n".join(parts)
    if missing_bodies:
        out += ("\n\n# not found in the source: " + ", ".join(missing_bodies)
                + " — fetch with read_source_functions_tool")
    return out or "(no bodies found — use read_source_functions_tool)"


def _compact_case_fact_status(state) -> dict:
    """Build a small status object for the model without carrying the full work-list.

    The complete migration lists live in migration_progress.json / the converted file.
    Keeping them out of ADK state prevents them from being echoed into every LLM turn.
    """
    existing = state.get("status") or {}
    converted_names = _converted_function_names(state)
    missing = [
        name for name in _source_function_names()
        if name not in converted_names
    ]
    # `constant_mismatch_sample` is the key the case-fact checker actually
    # writes. Reading the pre-trim name (`constant_value_mismatch`) yielded an
    # empty list on every turn, so the converter was told the constants were
    # fine no matter how many the checker had flagged — and constants are the
    # one thing the loop cannot discover for itself, because the checker
    # escalates only when they all match.
    mismatch = existing.get("constant_mismatch_sample") or []
    mismatch_count = existing.get("constant_value_mismatch_count", len(mismatch))
    return {
        "status": existing.get("status", "error"),
        "function_missing_count": len(missing),
        "function_missing_sample": missing[:20],
        "constant_value_mismatch": mismatch[:20],
        "constant_value_mismatch_count": mismatch_count,
        "constant_value_mismatch_truncated": mismatch_count > len(mismatch[:20]),
        "message": (
            "Convert the next batch of functions. The authoritative work-list "
            "is derived from the source and converted files on disk."
        ),
    }


def _converted_function_names(state) -> set[str]:
    """Return names already present in the assembled converted file."""
    p = _canonical_output_path(state)
    if not p.exists():
        return set()
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    return {
        n.name for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def seed_fact_status(callback_context: CallbackContext) -> None:
    """Prepare only compact, deterministic state for the converter turn."""
    state = callback_context.state
    state["status"] = _compact_case_fact_status(state)
    state["next_batch_source"] = _next_batch_source(state)
    return None


def seed_conventions(callback_context: CallbackContext) -> None:
    """Keep fixer state compact; conventions are supplied by the SkillToolset."""
    state = callback_context.state
    # Do not inject the complete SKILL.md/reference corpus into state.
    # The SkillToolset remains available to the fixer when it needs conventions.
    state.pop("pyspark_conventions", None)
    return None


HOST = os.environ["DATABRICKS_HOST"]
TOKEN = os.environ["DATABRICKS_API_KEY"]
USER=os.environ["USER_ID"]

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type" : "application/json"
}

def execute_pyspark_script_tool(context: ToolContext) -> dict:
    """Run the converted PySpark file on Databricks to catch syntax/runtime errors.

    Uploads the current converted file (path read from state) to the Databricks
    workspace as a notebook, submits it to serverless compute, waits for the run
    to finish, and deletes the uploaded notebook. Returns a compact dict so the
    agent can decide whether to fix and retry — the raw Databricks payload is
    NOT returned in full (it is large and would flood the context window).
    """
    python_script_path = context.state.get("converted_pyspark_file_path")
    if not python_script_path:
        return {
            "success": False,
            "error": "No converted PySpark file found in state. Call add_converted_functions_tool first.",
        }

    python_script_path = str(python_script_path)

    file_name = f"generated_{uuid.uuid4().hex}.py"
    workspace_path = f"/Workspace/Users/{USER}@shell.com/Drafts/{file_name}"
    # Read before the try block, so guard it separately: an unreadable file here
    # would otherwise escape as an exception instead of a tool result.
    try:
        with open(python_script_path, "r") as file:
            code = file.read()
    except OSError as exc:
        return {
            "success": False,
            "file_path": python_script_path,
            "error": f"Could not read the converted file: {exc}",
        }
    timeout = 600

    try:
        upload_payload = {
            "path":workspace_path,
            "format":"SOURCE",
            "language":"PYTHON",
            "overwrite":True,
            "content":base64.b64encode(code.encode()).decode()
        }

        r = requests.post(
            f"{HOST}/api/2.0/workspace/import",
            headers=HEADERS,
            json=upload_payload
        )

        if not r.ok:
            return {
                "success": False,
                "file_path": python_script_path,
                "status_code": r.status_code,
                "error": _tail(r.text, 1000),
            }
        r.raise_for_status()

        submit_payload = {
            "run_name":"varification",
            "tasks":[
                {
                    "task_key":"execute",
                    "notebook_task":{
                        "notebook_path":workspace_path,
                    },
                    "environment_key":"default_python"
                }

            ],
            "environments":[
                {
                    "environment_key":"default_python",
                    "spec":{
                        "environment_version":"4"
                    }
                }
            ]
        }

        r = requests.post(
            f"{HOST}/api/2.2/jobs/runs/submit",
            headers=HEADERS,
            json=submit_payload
        )

        r.raise_for_status()

        run_id = r.json()["run_id"]

        start = time.time()

        while True:
            r = requests.get(
                f"{HOST}/api/2.2/jobs/runs/get",
                headers=HEADERS,
                params={"run_id": run_id},
            )

            r.raise_for_status()
            info = r.json()
            task = info["tasks"][0]
            task_run_id = task["run_id"]
            state = task["state"]["life_cycle_state"]

            if state in ["TERMINATED", "INTERNAL_ERROR", "SKIPPED"]:
                break

            if time.time()-start>timeout:
                return {
                    "success": False,
                    "file_path": python_script_path,
                    "status": "TIMEOUT",
                    "run_id": run_id,
                    "life_cycle_state": state,
                }

            time.sleep(5)

        r = requests.get(
            f"{HOST}/api/2.2/jobs/runs/get-output",
            headers=HEADERS,
            params={"run_id":task_run_id}
        )
        output = r.json()

        status = info["state"].get("result_state")
        success = status == "SUCCESS"
        out = {
            "success": success,
            "file_path": python_script_path,
            "status": status,
            "run_id": run_id,
            "life_cycle_state": state,
        }
        if not success:
            error_text = output.get("error") or ""
            trace = output.get("error_trace") or ""
            out["error"] = _tail(error_text)
            out["error_summary"] = _error_summary(trace or error_text, python_script_path)
        return out

    except Exception as exc:
        return {
            "success": False,
            "file_path": python_script_path,
            "error": f"Databricks execution failed: {type(exc).__name__}: {exc}",
        }

    finally:
        try:
            requests.post(
                f"{HOST}/api/2.0/workspace/delete",
                headers=HEADERS,
                json={
                    "path":workspace_path,
                    "recursive":False,
                },
            )
            print("workspace_deleted")
        except Exception as ex:
            print("cleanup failed")


py_to_spark_skill = load_skill_from_dir(
    pathlib.Path(__file__).parent / "skills" / "py2snow-skill"
)

my_skill_toolset = SkillToolset(
    skills=[py_to_spark_skill]
)

code_convertor_agent = Agent(
    name="agent_code_converter",
    model = LiteLlm(
        model="databricks/databricks-claude-sonnet-4-6",
    ),
    instruction= """You are an expert coder who converts a Python ELT script into equivalent,
    distributed **PySpark** code. The output file is built up INCREMENTALLY across several
    turns — you convert a BATCH of functions each turn and APPEND them to the single output
    file. You must NEVER try to output the whole file at once (that truncates and produces
    "# continue similarly" stubs — which is a failure).

    MANDATORY conversion conventions: use the **py2snow-skill** through the SkillToolset
    for the native-Spark conversion rules. The skill MUST remain available and is the
    authoritative source for detailed conversion conventions. Use its resources when
    needed, but do not repeatedly reload or reread the same skill resource in one turn,
    and never reproduce the entire skill/reference corpus in the response. In particular,
    use native Spark APIs and avoid pandas idioms
    (`pd.`, `.merge`, `.rename(columns=...)`, `.iloc`, `df.apply`) and numpy column-building
    patterns unless the source is a deterministic data-generation function.

    THIS TURN'S BATCH — the original source of the next functions to convert is
    already here. Convert exactly these; do NOT call a tool to fetch them unless a
    body is actually missing or you need to verify a specific function:
    <batch_source>
    {next_batch_source}
    </batch_source>

    Other tools, only if you actually need them (each call costs a full round-trip):
      * **read_source_functions_tool(function_names=[...])** — ONLY if a body is
        missing from the supplied batch or you genuinely need to re-check it.
      * **read_source_index_tool()** — ONLY if you need metadata such as parameters
        or module constants that is not available from the current batch.
      * **read_migration_progress_tool()** — ONLY if progress is unclear.
      * **read_converted_file_tool()** — ONLY if you need output function names.
      * Do NOT repeatedly call a tool for information already present in this turn.
      * Do NOT repeatedly load/re-read the same skill resource in one turn. The
        py2snow-skill remains authoritative and available through SkillToolset.

    WORK-STATE (compact):
    <case_fact_status>
    {status}
    </case_fact_status>

    The complete work-list is derived from the source and converted files on disk.
    Do not ask for, reproduce, or store the whole work-list in state; convert only
    <batch_source>.

    You can use the **py2snow-skill** to guide each conversion.

    HOW TO WORK (batched append — follow exactly):
    1. Convert every function in <batch_source> above. Real, complete
       implementations: ABSOLUTELY NO placeholder comments (never write
       "# continue similarly", "# add remaining functions", "# TODO", "...", or an
       empty/`pass` body). A stubbed function does not count as converted and will
       just come back in the next work-list. If a batch is too long to finish inside
       your ~8k output limit, convert as many as fit and submit those — the rest
       return in the next work-list.
    3. Call **add_converted_functions_tool(functions_code=...)** passing ONLY this batch's
       new functions. On the FIRST batch also include: the needed `import` lines (pyspark
       imports, etc.) AND every module-level constant from the source with its EXACT value
       (`constant_value_mismatch` names the constants that are wrong;
       read_source_index_tool returns every constant name WITH its exact value —
       that tool is the only place the values come from, so use it instead of
       hunting for the raw source through the skill).
       Do NOT resend functions already in the file — the tool merges and de-dupes by name;
       it returns `converted_count`, `remaining_count`, and a small `remaining_sample`; use the count as the authoritative progress signal.
    4. STOP your turn as soon as the batch is appended. The loop re-invokes you with a freshly computed next batch; keep going batch by batch until `remaining_count` is 0.
    5. ONLY when `remaining_count` comes back 0 — the final batch — call
       **execute_pyspark_script_tool** once as a whole-file check. Do NOT run it after
       every batch: the converted file is a function library, so running it only
       proves the file imports, and each run costs a full round-trip. If it reports
       success=false, read `error_summary` and fix just the broken piece.

    **STRICT RULES**
    - Do not change the underlying logic of the functions.
    - Do not infer/rename columns — use the source script column names.
    - Do not change constant values; include every source constant with its correct value,
      and do NOT invent constants that are not in the source.
    - Convert same-named functions (the PySpark function name must equal the source name).
    - Convert EVERY source function, including the orchestrator (e.g. `run_all`) — it is a
      normal function. Do NOT invent helpers that are not in the source (the conventions
      skeleton already provides `get_spark`; do not add a `get_spark_session`).
    - IMPORTS: every non-local name you use MUST be imported — follow section "3a. Imports
      discipline" of the conventions above (name → import lookup). Re-check imports for each
      batch before submitting; a missing import becomes a NameError at test time.
    - SEMS / clean-code compliance is MANDATORY (see the "SEMS Compliance Conventions"
      section above): typed signatures, a docstring on every function naming its source
      function, comments on non-trivial transforms, `logging` (never `print`), specific
      `try/except` (never bare/broad, never `except: pass`) around risky ops, values from
      `CONFIG` (no magic literals), and no dead/commented-out code, TODOs, or stubs.
    - Never re-emit or overwrite the whole file; only append your new batch.
    - Do NOT emit module-level executable code, demo calls, or `if __name__ == "__main__"`
      blocks — the file is a function library (such lines are stripped anyway).
    - Data-generation functions (those using numpy / `random` / seeded generators to build
      synthetic rows) must build the data in PLAIN Python and pass it to
      `spark.createDataFrame(...)`. Do NOT translate seeded random generation into
      `F.rand()` column expressions — it changes behaviour and breaks determinism.
    """,

    tools=[
        my_skill_toolset,
        read_source_index_tool,
        read_source_functions_tool,
        read_migration_progress_tool,
        add_converted_functions_tool,
        read_converted_file_tool,
        execute_pyspark_script_tool,
    ],

    mode="task",
    output_key="code_converter_output",
    before_agent_callback=seed_fact_status,
)

code_fixer_agent = Agent(
    name="code_fixer_agent",
    model = LiteLlm(
        model="databricks/databricks-claude-opus-4-7",
    ),
    instruction= """You are an expert PySpark engineer. A converted PySpark pipeline
    already exists on disk (it may contain dozens of functions), but its pytest parity
    suite is FAILING. Fix ONLY the converted code — you do NOT edit the tests, and you
    fix ONLY the functions that are actually failing, one small batch at a time.

    MANDATORY conversion conventions: use the **py2snow-skill** through the SkillToolset
    for native Spark rules. Do not reproduce the full skill/reference corpus in context.
    Never introduce pandas idioms (`pd.`, `.merge`, `.rename(columns=)`, `.iloc`, `df.apply`)
    or numpy column-building patterns into the corrected code.

    Condensed pytest result — this lists only the FAILING tests and a short error for
    each (test `test_<function_name>` maps to the function `<function_name>`):
    <pytest_result>
    {pytest_last_stdout}
    </pytest_result>

    Latest parity verdict:
    <parity_test_status>
    {parity_test_status}
    </parity_test_status>

    HOW TO WORK (surgical, batched — follow exactly):
    1. From <pytest_result>, list the FAILING functions (strip the `test_` prefix from
       each failing test name). If nothing is failing, make NO changes and stop.
    2. Take the first few failing functions (up to 4). Call **read_functions_tool** with
       exactly those names to get their CURRENT (converted) source, AND
       **read_source_functions_tool** with the same names to get the ORIGINAL Python
       source. Do NOT pull the whole file.
    3. Compare the two. The original Python is the GROUND TRUTH for what the function must
       do — fix the converted function so its behaviour matches the original. If the short
       error references something that does not exist in the original (a column, a call, a
       line), that is a hallucination in the converted code — remove it.
    4. Call **replace_functions_tool(functions_code=...)** passing ONLY the corrected
       function(s) (plus any missing import/constant they need). It replaces those
       functions in place and leaves every other function untouched — NEVER paste the
       whole file, NEVER re-send functions you are not changing, NEVER add module-level
       calls or `if __name__ == "__main__"` blocks (they are stripped anyway).
    5. Call **execute_pyspark_script_tool** ONCE as a syntax/runtime sanity check.
    6. If more failing functions remain, repeat from step 2 for the next batch. Then stop
       — the parity agent re-runs the full suite and returns any still-failing functions.

    **STRICT RULES**
    - Do NOT change the underlying business logic; match the ORIGINAL Python behaviour.
    - Do NOT change or infer column names; do NOT change any constant values.
    - Data-generation functions (e.g. those using numpy / `random` / seeded generators to
      build synthetic rows) must build the data in PLAIN Python and pass it to
      `spark.createDataFrame(...)`. Do NOT translate seeded random generation into
      `F.rand()` column expressions — that changes behaviour and is a common wrong fix.
    - Keep the corrected function SEMS-compliant per the conventions above (typed signature,
      docstring, comments on non-trivial logic, `logging` not `print`, specific `try/except`
      never bare, values from `CONFIG`, no dead/commented-out code or TODOs).
    - The success signal for execute_pyspark_script_tool is the `success` field
      (`status == "SUCCESS"` on the Databricks run). Do NOT re-run when success is true.
      A `status` of "TIMEOUT" means the Databricks run did not finish in time — that is
      an infrastructure result, not evidence that a function is wrong.
    - You do NOT run the pytest suite yourself.
    """,

    tools=[
        my_skill_toolset,
        read_functions_tool,
        read_source_functions_tool,
        replace_functions_tool,
        execute_pyspark_script_tool,
    ],

    mode="task",
    output_key="code_fixer_output",
    before_agent_callback=seed_conventions,
)

semantic_code_fixer_agent = Agent(
    name="semantic_code_fixer_agent",
    model = LiteLlm(
        model="databricks/databricks-claude-sonnet-4-6",
    ),
    instruction= """You are an expert PySpark engineer. A converted PySpark pipeline
    exists on disk (possibly dozens of functions), but when the SAME data is run through
    both the source Python pipeline and the converted PySpark pipeline, their OUTPUTS DO
    NOT MATCH. Fix ONLY the converted PySpark code so its output equals the Python output,
    editing ONLY the functions responsible for the differences — surgically, in batches.

    MANDATORY conversion conventions: use the **py2snow-skill** through the SkillToolset
    for native Spark rules. Do not reproduce the full skill/reference corpus in context.
    Never introduce pandas idioms (`pd.`, `.merge`, `.rename(columns=)`, `.iloc`, `df.apply`)
    or numpy column-building patterns into the corrected code.

    Semantic comparison verdict (differences between the two outputs — these describe
    columns/values, so reason about WHICH function produces each differing column):
    <semantic_match>
    {semantic_match}
    </semantic_match>

    HOW TO WORK (surgical, batched — follow exactly):
    1. If `semantic_match.match` is already true, make NO changes and stop.
    2. From `semantic_match.differences`, work out the small set of functions responsible
       (the ones that compute/transform the differing columns). If you are unsure which
       functions exist, call **read_converted_file_tool** ONCE — it lists names only.
    3. For those functions, call **read_functions_tool** (current converted source) AND
       **read_source_functions_tool** (ORIGINAL Python source = ground truth). Do NOT pull
       the whole file if you only need a few functions.
    4. Diagnose each difference against the ORIGINAL Python behaviour (wrong join type,
       aggregation, window/ordering semantics, type/precision, null-handling, column
       names) and correct only those functions so they reproduce the Python result.
    5. Call **replace_functions_tool(functions_code=...)** with ONLY the corrected
       function(s). It replaces them in place and leaves every other function untouched —
       NEVER paste the whole file, NEVER re-send functions you are not changing, NEVER add
       module-level calls or `if __name__ == "__main__"` blocks (they are stripped anyway).
    6. Call **execute_pyspark_script_tool** ONCE as a syntax/runtime sanity check, then stop.
       The semantic agent re-runs both pipelines and returns a fresh diff if needed.

    **STRICT RULES**
    - Fix ONLY the converted PySpark code. Do NOT change the source Python, the dummy
      dataset, the runner scripts, or the recorded Python output.
    - Do NOT change constant values or the intended business logic; match the ORIGINAL
      Python behaviour so the outputs are equal.
    - Data-generation functions (numpy / `random` / seeded generators) must build data in
      PLAIN Python and use `spark.createDataFrame(...)`; do NOT translate seeded random
      generation into `F.rand()` column expressions.
    - Keep the corrected function SEMS-compliant per the conventions above (typed signature,
      docstring, comments on non-trivial logic, `logging` not `print`, specific `try/except`
      never bare, values from `CONFIG`, no dead/commented-out code or TODOs).
    - The success signal for execute_pyspark_script_tool is the `success` field
      (`status == "SUCCESS"` on the Databricks run). Do NOT re-run when success is true.
      A `status` of "TIMEOUT" means the Databricks run did not finish in time — that is
      an infrastructure result, not evidence that a function is wrong.
    """,

    tools=[
        my_skill_toolset,
        read_functions_tool,
        read_source_functions_tool,
        read_converted_file_tool,
        replace_functions_tool,
        execute_pyspark_script_tool,
    ],

    mode="task",
    output_key="semantic_code_fixer_output",
    before_agent_callback=seed_conventions,
)