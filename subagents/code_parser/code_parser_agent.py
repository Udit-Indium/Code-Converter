import os
import ast
import json
import re
from pathlib import Path
import nbformat
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.adk import Agent
from dotenv import load_dotenv
from .scripts.ast_parser import run_parser
from ..model_config import CODE_PARSER_MODEL
load_dotenv()

def _safe_stem(name: str) -> str:
    """Make a filename stem usable as a Python module name."""
    stem = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()
    if not stem:
        stem = "source_script"
    if stem[0].isdigit():
        stem = f"s_{stem}"
    return stem


_NOTEBOOK_ONLY = (
    re.compile(r"^\s*[\w.\[\]'\"]+\s*=\s*!"),
    re.compile(r"^\s*[\w.()\[\]]+\?\??\s*$"),
    re.compile(r"^await\s+"),
)


def _is_notebook_only(line: str) -> bool:
    """True if `line` is notebook syntax that is not valid Python in a module.

    Covers line/cell magics (`%pip`, `%%sql`), shell escapes (`!pip`), the
    shell-capture assignment form (`files = !ls`), the `?`/`??` help suffix, and
    a top-level `await`. All of these run fine in a notebook and are a
    SyntaxError once the cells are flattened into a module.
    """
    stripped = line.lstrip()
    if stripped.startswith(("%", "!", "?")):
        return True
    if stripped.startswith("dbutils.notebook.exit"):
        return True
    return any(p.match(line) for p in _NOTEBOOK_ONLY)


def _parse_report(text: str) -> tuple[bool, str, str, int]:
    """Try to parse `text`; on failure return the error, a context snippet, and
    the offending line number.

    The line number is returned as an int rather than left for the caller to dig
    back out of the message string — SyntaxError messages contain colons and
    spaces of their own, so re-parsing the formatted text is unreliable.

    A bare "line 41: invalid syntax" is also hard to act on when the file was
    assembled from cells, hence the snippet.
    """
    try:
        ast.parse(text)
        return True, "", "", 0
    except SyntaxError as exc:
        n = exc.lineno or 0
        err = f"line {n}: {exc.msg}"
        rows = text.splitlines()
        if not (1 <= n <= len(rows)):
            return False, err, "", 0
        lo, hi = max(1, n - 3), min(len(rows), n + 3)
        snippet = "\n".join(
            f"{'>>' if i == n else '  '} {i:>4} | {rows[i - 1]}"
            for i in range(lo, hi + 1)
        )
        return False, err, snippet, n


def notebook_to_python(context: ToolContext, script_path: str) -> dict[str, str]:
    """Convert a Jupyter/Databricks notebook (.ipynb) into a plain .py script.

    Call this FIRST, before ast_parser. The parser uses Python's `ast`, which
    cannot read a notebook's JSON — so a .ipynb has to be flattened to source
    first.

    Reading goes through `nbformat`, so older notebook formats are upgraded to
    v4 before the cells are walked. The flattening itself is deliberately ours
    rather than `nbconvert`'s: cells stay in place with `# --- code cell N ---`
    markers, markdown is preserved as comments, and notebook-only syntax is
    commented rather than dropped — all of which the later agents rely on to
    trace generated code back to the notebook.

    If `script_path` is already a .py file this is a no-op: it reports
    `converted: false` and hands the same path back, so it is always safe to call.

    Args:
        context: agent state.
        script_path: path to the .ipynb (or .py) source.

    Returns:
        `python_script_path` — the path every later tool should use.
    """
    src = Path(script_path)

    if src.suffix.lower() != ".ipynb":
        return {
            "converted": False,
            "python_script_path": str(src),
            "message": f"'{src.name}' is already a Python file; no conversion needed.",
        }
    try:
        notebook = nbformat.read(str(src), as_version=4)
    except (OSError, ValueError, nbformat.ValidationError) as exc:
        return {
            "converted": False,
            "message": f"Could not read '{script_path}' as a notebook: {exc}",
        }
    schema_warning = ""
    try:
        nbformat.validate(notebook)
    except nbformat.ValidationError as exc:
        schema_warning = str(exc).splitlines()[0]

    language = (
        notebook.get("metadata", {}).get("kernelspec", {}).get("language", "")
        or notebook.get("metadata", {}).get("language_info", {}).get("name", "")
    )

    lines: list[str] = [f'"""Flattened from {src.name}."""', ""]
    code_cells = 0
    magics: list[str] = []

    for i, cell in enumerate(notebook.cells, start=1):
        kind = cell.get("cell_type")
        body = (cell.get("source") or "").rstrip("\n")
        if not body.strip():
            continue

        if kind == "markdown":
            lines.append(f"# --- markdown cell {i} ---")
            lines.extend(f"# {ln}" for ln in body.splitlines())
            lines.append("")
            continue

        if kind != "code":
            continue

        code_cells += 1
        lines.append(f"# --- code cell {i} ---")
        for ln in body.splitlines():
            if _is_notebook_only(ln):
                magics.append(f"cell {i}: {ln.strip()}")
                lines.append(f"# [notebook-only] {ln}")
            else:
                lines.append(ln)
        lines.append("")

    if code_cells == 0:
        return {
            "converted": False,
            "message": f"'{script_path}' contains no code cells — nothing to convert.",
        }

    output_dir = Path(__file__).parent.parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{_safe_stem(src.stem)}.py"
    text = "\n".join(lines).rstrip() + "\n"
    parses, parse_error, error_context, bad_line = _parse_report(text)
    auto_commented: list[str] = []
    for _ in range(10):
        if parses:
            break
        rows = text.splitlines()
        if not (1 <= bad_line <= len(rows)) or rows[bad_line - 1].lstrip().startswith("#"):
            break
        auto_commented.append(f"line {bad_line}: {rows[bad_line - 1].strip()}")
        rows[bad_line - 1] = f"# [unparseable] {rows[bad_line - 1]}"
        text = "\n".join(rows) + "\n"
        parses, parse_error, error_context, bad_line = _parse_report(text)

    # Rewrite only when the content actually differs. This is what makes the
    # refactored file's staleness check meaningful downstream: this function
    # runs again on every pipeline pass, and an unconditional write would bump
    # the flat script's mtime past the refactored script derived from it,
    # making a perfectly current refactor look stale on every single run.
    unchanged = False
    try:
        unchanged = out_path.is_file() and out_path.read_text(encoding="utf-8") == text
    except OSError:
        unchanged = False

    if not unchanged:
        try:
            out_path.write_text(text, encoding="utf-8")
        except OSError as exc:
            return {"converted": False, "message": f"Could not write '{out_path}': {exc}"}

    # Nothing is written to state: the flattened file IS the handoff, and a
    # state key here would be overwritten on every re-run, clobbering whatever
    # a later stage had put there.
    result = {
        "converted": True,
        "python_script_path": str(out_path),
        "code_cells": code_cells,
        "parses_as_python": parses,
        "message": (
            f"Converted {code_cells} code cell(s) from '{src.name}' to '{out_path}'. "
            "Use python_script_path for ast_parser."
        ),
    }
    if language and language.lower() not in ("python", "python3"):
        result["kernel_language"] = language
        result["message"] += (
            f" WARNING: the notebook's kernel language is '{language}', not Python."
            " The flattened cells are unlikely to be valid Python."
        )
    if schema_warning:
        result["schema_warning"] = schema_warning
    if magics:
        result["commented_out_notebook_lines"] = magics[:20]
    if auto_commented:
        result["auto_commented_unparseable_lines"] = auto_commented
        result["message"] += (
            f" NOTE: {len(auto_commented)} line(s) would not parse as Python and were"
            " commented out — see auto_commented_unparseable_lines. If any of them was"
            " real logic, the conversion will be incomplete."
        )
    if not parses:
        result["parse_error"] = parse_error
        result["error_context"] = error_context
        result["message"] += (
            f" WARNING: the flattened file still does not parse ({parse_error}). "
            "See error_context for the offending line. The usual cause is a code "
            "block split across two cells, which cannot be fixed by commenting a "
            "single line. ast_parser will produce an EMPTY inventory until it is fixed."
        )
    return result


#: Everything the pipeline writes lands here.
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"

#: Canonical name for the parsed inventory. Fixed rather than derived from the
#: input filename so a consumer needs no knowledge of what was parsed.
AST_INVENTORY = OUTPUT_DIR / "ast_inventory.json"

#: Suffix the refactor stage appends to the script it restructures.
_REFACTORED_SUFFIX = "_refactored"


def _find_refactored(script_path: Path) -> tuple[Path | None, str]:
    """Return the refactor stage's output for `script_path`, if it is usable.

    This is a naming convention rather than a lookup in agent state, and that
    is deliberate: `notebook_to_python` rewrites its own state key every time it
    runs, so a state-based handoff let a later stage silently overwrite the
    refactored path and undo the refactor. A file either exists on disk or it
    does not.

    A refactored file OLDER than the script it was derived from is left behind
    by a previous run against different source. Using it would convert stale
    code and say nothing, so it is rejected — but loudly, via the returned
    note, because the alternative (parsing the flat script) yields a nearly
    empty inventory and the operator needs to know why.

    Returns:
        `(path, note)`. `path` is None when there is no usable refactored
        version — the normal case for a script that was already modular.
        `note` is non-empty only when something needs saying.
    """
    # Already the refactored file: do not look for `..._refactored_refactored`.
    if script_path.stem.endswith(_REFACTORED_SUFFIX):
        return (script_path if script_path.is_file() else None), ""

    candidate = OUTPUT_DIR / f"{script_path.stem}{_REFACTORED_SUFFIX}.py"
    if not candidate.is_file():
        return None, ""

    try:
        if candidate.stat().st_mtime < script_path.stat().st_mtime:
            return None, (
                f"IGNORED a stale refactored script: '{candidate.name}' is older "
                f"than the source it came from ('{script_path.name}'), so it "
                "reflects a previous version. Parsing the flat script instead, "
                "which has few top-level functions — re-run the refactor stage "
                "to fix this."
            )
    except OSError:
        # Cannot compare timestamps; prefer the refactored file, since that is
        # the case this whole lookup exists to serve.
        return candidate, ""

    return candidate, ""


def _resolve_python_path(
    context: ToolContext, script_path: str
) -> tuple[str, str, str]:
    """Return a .py path for `script_path`, converting a notebook if needed.

    Guards both parser tools against being handed a .ipynb. That has to be caught
    here rather than left to the prompt, because it fails SILENTLY: a notebook's
    JSON (`{"cells": [...]}`) is a valid Python dict literal, so `ast.parse`
    succeeds and returns an empty inventory — no functions, no constants, no
    error. The conversion loop would then run to max_iterations with nothing to
    convert and no indication why.

    Returns:
        `(path, error, note)`. `path` and `error` are mutually exclusive.
        `note` carries a non-fatal warning worth surfacing, such as a stale
        refactored script having been ignored.
    """
    src = Path(script_path)

    # A notebook has to be flattened before anything can look at it as Python.
    if src.suffix.lower() == ".ipynb":
        result = notebook_to_python(context, script_path)
        if not (result.get("converted") and result.get("python_script_path")):
            return "", result.get(
                "message", f"could not convert notebook '{script_path}'"
            ), ""
        src = Path(result["python_script_path"])

    # Prefer the refactor stage's output, found by naming convention rather
    # than through state. The flat script has almost no top-level functions and
    # the converter works function by function, so parsing the flat version
    # when a refactored one exists would hand the converter an empty inventory
    # and no explanation.
    refactored, note = _find_refactored(src)
    return str(refactored or src), "", note


def ast_parser(context: ToolContext, script_path: str)-> dict[str, str]:
    """
    This tool will be used to parse the python script using ast parser

    Parses the REFACTORED script when the refactor stage produced one, found by
    naming convention rather than through state.

    The parsed inventory is written to `outputs/ast_inventory.json` and NOT put
    in state. For a script of a few hundred functions the inventory is tens of
    kilobytes of JSON, and everything in state is echoed into every prompt of
    every agent sharing the session. Downstream agents load it from that file.

    Args:
    context :- The state of the agent of type ToolContext
    script_path :- the path of the script.
    """
    script_path, err, note = _resolve_python_path(context, script_path)
    if err:
        return {"message": f"cannot parse: {err}"}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = run_parser(script_path, follow_imports=True, output_dir=str(OUTPUT_DIR))
        with open(script_path, "r") as file:
            python_content = file.read()

        with open(result["json_file"], "r") as file:
            ast_parsed_content = json.load(file)

        # Two canonical copies so consumers need no knowledge of the original
        # filename, and no path has to travel in state.
        source_copy = OUTPUT_DIR / "source_script.py"
        source_copy.write_text(python_content, encoding="utf-8")
        AST_INVENTORY.write_text(
            json.dumps(ast_parsed_content, indent=2, default=str), encoding="utf-8"
        )

        functions = ast_parsed_content.get("functions") or []
        message = (
            f"file has been successfully parsed: {len(functions)} function(s) "
            f"found in '{Path(script_path).name}'."
        )
        out = {
            "message": f"{message} WARNING: {note}" if note else message,
            "parsed_script_path": str(script_path),
            "source_script_path": str(source_copy),
            "ast_parsed_json_path": str(AST_INVENTORY),
            "function_count": len(functions),
        }
        if note:
            out["warning"] = note
        return out
    except Exception as e:
        return {
            "message": f"there is some error while parsing the python file usinf ast refer this: {e}"
        }


code_parser_agent = Agent(
    model = LiteLlm(
        model=f"databricks/{CODE_PARSER_MODEL}",
    ),
    name="code_parser_agent",
    instruction="""
    You are a helpful code parser which parses the source script using the available tools.

    Tools:
    notebook_to_python :- flatten a Jupyter/Databricks notebook (.ipynb) into a plain
      .py script. Returns `python_script_path`.
    ast_parser :- parse the python file using the ast parser.

    HOW TO WORK:
    1. If the given path ends in `.ipynb`, call **notebook_to_python** FIRST and wait
       for it. ast_parser uses Python's `ast`, which cannot read a notebook's JSON —
       it will fail on a .ipynb. Use the `python_script_path` it returns for step 2.
       If the path already ends in `.py`, skip this step and use the path as given.
    2. Call **ast_parser** on that path.
    3. If notebook_to_python reports `parses_as_python: false`, say so plainly and
       name the offending line instead of proceeding — the parse will fail otherwise.
    """,
    tools=[
            notebook_to_python,
            ast_parser
    ],
    mode="single_turn"
)