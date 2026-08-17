import os
import ast
import base64
import json
import pathlib
import time
import uuid
import requests
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from dotenv import load_dotenv
from .tools import run_parser


from ...model_config import PARITY_MODEL
load_dotenv()

OUTPUTS_DIR = pathlib.Path(__file__).parents[3] / "outputs"
PYTEST_FILENAME = "pyspark_pytest.py"

HOST = os.environ["DATABRICKS_HOST"]
TOKEN = os.environ["DATABRICKS_API_KEY"]
USER = os.environ["USER_ID"]

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}


def _pytest_path() -> pathlib.Path:
    return OUTPUTS_DIR / PYTEST_FILENAME


def load_functions_to_test(callback_context: CallbackContext) -> None:
    """Pre-agent-call callback.

    Reads the converted PySpark script path from state, parses it, and stores
    the list of function names that the agent must generate test cases for
    under state["functions_to_test"]. That is all this callback does — it does
    NOT look at tests or coverage.
    """
    state = callback_context.state
    state.setdefault("pytest_last_stdout", "")
    state.setdefault("pytest_last_stderr", "")
    state.setdefault("pytest_last_returncode", None)

    script_path = state.get("converted_pyspark_file_path")
    if not script_path:
        state["functions_to_test"] = {
            "count": 0,
            "names": [],
            "error": "No 'converted_pyspark_file_path' found in state.",
        }
        state.setdefault("parity_test_status", {
            "status": "false",
            "missing_functions": [],
            "message": "No 'converted_pyspark_file_path' found in state.",
        })
        return

    try:
        parsed = run_parser(script_path, follow_imports=False)
    except Exception as exc:
        state["functions_to_test"] = {
            "count": 0,
            "names": [],
            "error": f"Failed to parse PySpark script '{script_path}': {exc}",
        }
        state.setdefault("parity_test_status", {
            "status": "false",
            "missing_functions": [],
            "message": f"Failed to parse PySpark script '{script_path}': {exc}",
        })
        return

    functions = parsed.get("functions") or []
    function_names = [f.get("name") for f in functions if isinstance(f, dict) and f.get("name")]

    state["functions_to_test"] = {
        "count": len(function_names),
        "names": function_names,
    }

    state["pyspark_module_name"] = pathlib.Path(script_path).stem

    state.setdefault("parity_test_status", {
        "status": "false",
        "missing_functions": function_names,
        "message": (
            "No test cases generated yet; generate a test_<function_name> "
            "for every function."
        ),
    })

    return None

def _extract_test_functions(test_source: str) -> list[str]:
    """Return the names of every `def test_*` defined in the test file."""
    if not test_source.strip():
        return []
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return []
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    ]

def _missing_functions(target_names: list[str], test_functions: list[str]) -> list[str]:
    """Target functions with no test of their own.

    Each test is attributed to the LONGEST target name it matches, and to that
    one only. That specificity matters because function names routinely share a
    prefix — the refactor stage disambiguates repeats as `combine_query_sf`,
    `combine_query_sf_2`, `combine_query_sf_3`. A looser rule ("any test whose
    name contains the function name") lets `test_combine_query_sf_4` count as
    the test for `combine_query_sf` as well, so one test silently satisfies a
    whole family and the agent's one non-negotiable rule — a test per function —
    is reported as met when it is not.

    Still counts a suffixed variant: `test_load_orders_handles_nulls` covers
    `load_orders`, as long as no longer target name matches it better.
    """
    by_specificity = sorted(set(target_names), key=len, reverse=True)
    covered: set[str] = set()

    for test_name in test_functions:
        if not test_name.startswith("test_"):
            continue
        stem = test_name[len("test_"):]
        for fn in by_specificity:
            if stem == fn or stem.startswith(f"{fn}_"):
                covered.add(fn)
                break  # most specific target wins; do not credit its prefixes

    return [fn for fn in target_names if fn not in covered]


_MAX_ERR_CHARS = 300     
_MAX_SUMMARY_CHARS = 6000 


def _cap(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def _summarize_pytest(stdout: str, stderr: str, returncode: "int | None") -> str:
    """Condense raw pytest output down to only what the code-fixer needs.

    PySpark failures embed enormous Py4J/Scala Java stack traces that otherwise
    flood the LLM context. We keep just: the counts line, and one line per
    failing/erroring test (its node id + the first line of the error, capped).
    """
    lines = (stdout or "").splitlines()
    counts = ""
    for l in reversed(lines):
        s = l.strip()
        if "====" in s and ("passed" in s or "failed" in s or "error" in s):
            counts = s.strip("= ").strip()
            break
    failures: list[str] = []
    for l in lines:
        s = l.strip()
        if s.startswith("FAILED ") or s.startswith("ERROR "):
            failures.append(_cap(s, _MAX_ERR_CHARS + 80))

    if returncode == 0 and not failures:
        return f"All tests passed. ({counts})" if counts else "All tests passed."

    parts: list[str] = []
    if counts:
        parts.append(counts)
    if failures:
        parts.append(f"{len(failures)} failing test(s):")
        parts.extend(f"  - {f}" for f in failures)
    else:
        parts.append("No per-test summary parsed; tail of stderr:")
        parts.append(_cap(stderr or stdout, 1500))

    return "\n".join(parts)[:_MAX_SUMMARY_CHARS]


_PYTEST_DRIVER_BODY = '''
import base64, contextlib, io, json, os, sys

_DIR = "/tmp/parity_suite"
os.makedirs(_DIR, exist_ok=True)

with open(os.path.join(_DIR, _MODULE_NAME + ".py"), "wb") as _f:
    _f.write(base64.b64decode(_MODULE_B64))
with open(os.path.join(_DIR, "pyspark_pytest.py"), "wb") as _f:
    _f.write(base64.b64decode(_TEST_B64))

if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
os.chdir(_DIR)

try:
    import pytest
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=False)
    import pytest

_buf = io.StringIO()
with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
    # -q quiet, --tb=line one-line tracebacks, -rfE short summary of failed +
    # errored tests. Together this keeps output tiny.
    _rc = pytest.main(["pyspark_pytest.py", "-q", "--tb=line", "-rfE", "-p", "no:cacheprovider"])

_out = _buf.getvalue()[-40000:]
dbutils.notebook.exit(json.dumps({"returncode": int(_rc), "stdout": _out}))
'''


def _pytest_driver_source(module_name: str, module_src: str, test_src: str) -> str:
    """Build the notebook source that materialises both files and runs pytest.

    Both payloads are embedded base64-encoded so arbitrary quotes, backslashes
    and newlines in the generated code cannot break out of the notebook source.
    """
    module_b64 = base64.b64encode(module_src.encode("utf-8")).decode()
    test_b64 = base64.b64encode(test_src.encode("utf-8")).decode()
    header = (
        f"_MODULE_NAME = {module_name!r}\n"
        f"_MODULE_B64 = {module_b64!r}\n"
        f"_TEST_B64 = {test_b64!r}\n"
    )
    return header + _PYTEST_DRIVER_BODY


def _run_pytest_suite(state) -> "int | None":
    """Run pyspark_pytest.py on Databricks and record the result in state.

    Uploads a driver notebook carrying BOTH the test suite and the converted
    PySpark module (the suite imports it), submits it to serverless compute,
    waits for the run, and reads pytest's returncode + stdout back out of the
    notebook exit value. The uploaded notebook is always deleted afterwards.

    Writes pytest_last_returncode / pytest_last_stdout / pytest_last_stderr and
    returns the returncode (None if a file is missing or the run did not
    produce a result). Shared by run_pytest_tool (agent-facing) and
    check_test_case_status (the authoritative run) so the recorded result always
    matches the files on disk.

    Only a CONDENSED failure summary is stored (failing tests + short error
    each) — never the raw multi-thousand-line Py4J/Scala traces, which would
    blow the LLM context window.
    """
    path = _pytest_path()
    if not path.exists():
        return None

    module_path = state.get("converted_pyspark_file_path")
    if not module_path or not os.path.isfile(str(module_path)):
        state["pytest_last_returncode"] = None
        state["pytest_last_stdout"] = ""
        state["pytest_last_stderr"] = (
            f"Converted PySpark module not found at {module_path!r} — cannot run the suite."
        )
        return None

    module_path = pathlib.Path(str(module_path))
    module_name = module_path.stem

    driver = _pytest_driver_source(
        module_name,
        module_path.read_text(encoding="utf-8"),
        path.read_text(encoding="utf-8"),
    )

    file_name = f"parity_{uuid.uuid4().hex}.py"
    workspace_path = f"/Workspace/Users/{USER}@shell.com/Drafts/{file_name}"
    timeout = 600

    try:
        r = requests.post(
            f"{HOST}/api/2.0/workspace/import",
            headers=HEADERS,
            json={
                "path": workspace_path,
                "format": "SOURCE",
                "language": "PYTHON",
                "overwrite": True,
                "content": base64.b64encode(driver.encode()).decode(),
            },
        )
        r.raise_for_status()

        r = requests.post(
            f"{HOST}/api/2.2/jobs/runs/submit",
            headers=HEADERS,
            json={
                "run_name": "parity_pytest",
                "tasks": [
                    {
                        "task_key": "pytest",
                        "notebook_task": {
                            "notebook_path": workspace_path,
                        },
                        "environment_key": "default_python",
                    }
                ],
                "environments": [
                    {
                        "environment_key": "default_python",
                        "spec": {
                            "environment_version": "4"
                        },
                    }
                ],
            },
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
            state_name = task["state"]["life_cycle_state"]

            if state_name in ["TERMINATED", "INTERNAL_ERROR", "SKIPPED"]:
                break

            if time.time() - start > timeout:
                state["pytest_last_returncode"] = None
                state["pytest_last_stdout"] = ""
                state["pytest_last_stderr"] = (
                    f"pytest run timed out after {timeout} seconds (run_id {run_id})."
                )
                return None

            time.sleep(5)

        r = requests.get(
            f"{HOST}/api/2.2/jobs/runs/get-output",
            headers=HEADERS,
            params={"run_id": task_run_id},
        )
        r.raise_for_status()
        output = r.json()

        result = (output.get("notebook_output") or {}).get("result")
        if not result:
            state["pytest_last_returncode"] = None
            state["pytest_last_stdout"] = ""
            state["pytest_last_stderr"] = _cap(
                "The pytest driver notebook did not return a result. "
                f"Databricks error: {output.get('error') or 'unknown'}",
                1500,
            )
            return None

        payload = json.loads(result)
        returncode = payload.get("returncode")
        stdout = payload.get("stdout") or ""

        state["pytest_last_returncode"] = returncode
        state["pytest_last_stdout"] = _summarize_pytest(stdout, "", returncode)
        state["pytest_last_stderr"] = "" 
        return returncode

    except Exception as exc:
        state["pytest_last_returncode"] = None
        state["pytest_last_stdout"] = ""
        state["pytest_last_stderr"] = (
            f"Databricks pytest run failed: {type(exc).__name__}: {exc}"
        )
        return None

    finally:
        try:
            requests.post(
                f"{HOST}/api/2.0/workspace/delete",
                headers=HEADERS,
                json={"path": workspace_path, "recursive": False},
            )
        except Exception:
            pass


def check_test_case_status(callback_context: CallbackContext) -> None:
    """Escalation criteria for the enclosing test-correction LoopAgent.

    Reads pyspark_pytest.py, extracts the test functions, compares them against
    the functions_to_test list, and combines that with the pytest result.

    Escalate (stop the loop) only when BOTH hold:
      * every target function has a test  (full coverage), AND
      * a fresh Databricks pytest run of the CURRENT file exits 0 (suite passes).

    The pytest result is (re)computed HERE on the final test file rather than
    trusting whatever the agent last ran — the agent may have run the suite
    before writing all tests, or added tests after its last run, leaving a stale
    returncode in state. Re-running guarantees the verdict matches the file.

    When coverage is complete but the suite fails, we do NOT escalate — the
    failure is likely a bug in the converted PySpark code, which the next
    sub-agent (code_fixer_agent) repairs before the loop re-tests. The verdict
    is written to state["parity_test_status"].
    """
    state = callback_context.state

    target_names = list((state.get("functions_to_test") or {}).get("names") or [])

    path = _pytest_path()
    test_source = path.read_text(encoding="utf-8") if path.exists() else ""
    test_functions = _extract_test_functions(test_source)

    missing = _missing_functions(target_names, test_functions)
    all_present = bool(target_names) and not missing
    tests_pass = False
    if all_present:
        tests_pass = _run_pytest_suite(state) == 0

    if all_present and tests_pass:
        state["parity_test_status"] = {
            "status": "success",
            "missing_functions": [],
            "message": (
                f"All {len(target_names)} test cases are present and the suite passes."
            ),
        }
        state["parity_validation_passed"] = True
        callback_context.actions.escalate = True
    elif not all_present:
        state["parity_test_status"] = {
            "status": "false",
            "missing_functions": missing,
            "message": (
                "Test cases are missing for the following functions; "
                "generate a test_<function_name> for each of them."
            ),
        }
        state["parity_validation_passed"] = False
    else:
        state["parity_test_status"] = {
            "status": "false",
            "missing_functions": [],
            "message": (
                "All functions have tests, but the suite is failing. The converted "
                "PySpark code likely needs to be fixed against the pytest errors."
            ),
        }
        state["parity_validation_passed"] = False

    return None

def read_converted_index_tool(context: ToolContext) -> dict:
    """Signatures of every function in the converted module — no bodies.

    The cheap overview: what exists, what each one takes and returns. Use it to
    plan your batches, then call read_converted_functions_tool for the bodies of
    just the handful you are writing tests for this turn.
    """
    path = context.state.get("converted_pyspark_file_path")
    if not path or not os.path.isfile(str(path)):
        return {"exists": False, "functions": [], "error": "no converted module yet"}
    try:
        tree = ast.parse(pathlib.Path(str(path)).read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        return {"exists": False, "functions": [], "error": f"could not parse: {exc}"}

    out = []
    for n in tree.body:
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        entry = {"name": n.name, "kind": type(n).__name__}
        if not isinstance(n, ast.ClassDef):
            entry["parameters"] = [a.arg for a in n.args.args]
            entry["returns"] = ast.unparse(n.returns) if n.returns else None
            entry["lines"] = (getattr(n, "end_lineno", n.lineno) or n.lineno) - n.lineno + 1
        out.append(entry)
    return {"exists": True, "count": len(out), "functions": out}


def read_converted_functions_tool(context: ToolContext, function_names: list[str]) -> dict:
    """Source of ONLY the named functions from the converted PySpark module.

    This is what you assert against — read the real body before writing a test,
    never guess from the name. Ask for the batch you are working on, not the
    whole module. Unknown names come back under `not_found`.
    """
    path = context.state.get("converted_pyspark_file_path")
    if not path or not os.path.isfile(str(path)):
        return {"exists": False, "functions": {},
                "not_found": list(function_names or [])}
    try:
        src = pathlib.Path(str(path)).read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (OSError, SyntaxError) as exc:
        return {"exists": False, "functions": {},
                "not_found": list(function_names or []), "error": str(exc)}

    by_name = {
        n.name: (ast.get_source_segment(src, n) or ast.unparse(n))
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    requested = list(function_names or [])
    return {
        "exists": True,
        "functions": {n: by_name[n] for n in requested if n in by_name},
        "not_found": [n for n in requested if n not in by_name],
    }


def _test_segments(src: str):
    """Split module source into top-level segments, keeping each one's ORIGINAL
    text. ast is used only to locate line spans, so comments and formatting in
    the generated tests survive the merge.

    Yields (kind, key, raw_text) with kind 'import' | 'def' | 'other'.
    """
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    prev_end = 0
    for node in tree.body:
        start = node.lineno
        if getattr(node, "decorator_list", None):          # @pytest.fixture
            start = min(d.lineno for d in node.decorator_list)
        top = start - 1
        while top - 1 >= prev_end and lines[top - 1].strip().startswith("#"):
            top -= 1
        raw = "".join(lines[top:node.end_lineno]).rstrip("\n")
        prev_end = node.end_lineno

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield "import", ast.unparse(node), raw
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield "def", node.name, raw
        else:
            yield "other", ast.unparse(node), raw


def _merge_tests(existing_src: str, snippet: str) -> tuple[str, list[str]]:
    """Merge a BATCH of tests into the existing suite, deterministically.

    The serving endpoint caps a single response at ~8k output tokens, so a full
    suite cannot be emitted in one tool call — generation is cut off mid-argument
    and ADK receives a function call with no arguments at all. Batching sidesteps
    that: each call carries a handful of tests and the FILE is rebuilt here in
    Python, so it can never truncate however many tests accumulate. Same approach
    as _assemble() in the converter.
    """
    import_order: list[str] = []
    imports: dict[str, str] = {}
    body_order: list[tuple] = []
    body: dict[tuple, str] = {}
    changed: list[str] = []

    def _ingest(src: str, is_snippet: bool):
        if not src.strip():
            return
        for kind, key, raw in _test_segments(src):
            if kind == "import":
                if key not in imports:
                    import_order.append(key)
                    imports[key] = raw
            else:
                bk = (kind, key)
                if bk not in body:
                    body_order.append(bk)
                    body[bk] = raw
                    if is_snippet and kind == "def":
                        changed.append(key)
                elif is_snippet:
                    body[bk] = raw               # overwrite, keep position
                    if kind == "def" and key not in changed:
                        changed.append(key)

    _ingest(existing_src, False)
    _ingest(snippet, True)

    parts = []
    if import_order:
        parts.append("\n".join(imports[k] for k in import_order))
    if body_order:
        parts.append("\n\n\n".join(body[bk] for bk in body_order))
    return "\n\n".join(parts).rstrip() + "\n", changed


def add_pytest_tests_tool(context: ToolContext, tests_code: str) -> dict:
    """Add a BATCH of tests to outputs/pyspark_pytest.py.

    Send ONLY the tests you wrote this turn — about 8 to 10 at a time, NEVER the
    whole suite. Your response is capped at roughly 8k tokens; a full suite does
    not fit.
    Args:
        tests_code: source for THIS BATCH only (plus imports/fixture on the first call).

    Returns:
        {"status", "saved_file_path", "tests_added_this_batch",
         "total_tests_in_file", "test_names"}
    """
    path = _pytest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    try:
        merged, changed = _merge_tests(existing, tests_code)
    except SyntaxError as exc:
        return {
            "status": "error",
            "error": f"The submitted tests_code has a syntax error: {exc}. "
                     "Fix it and resubmit only that batch.",
        }
    path.write_text(merged, encoding="utf-8")
    context.state["pyspark_pytest_file_path"] = str(path)

    total = _extract_test_functions(merged)
    return {
        "status": "success",
        "saved_file_path": str(path),
        "tests_added_this_batch": changed,
        "total_tests_in_file": len(total),
        "test_names": sorted(total),
    }


def read_pytest_file_tool(context: ToolContext) -> dict:
    """List the tests already written to outputs/pyspark_pytest.py.

    Returns test names and a line count, NOT the test code. The reason to call
    this is to avoid writing a test that already exists, and the names answer
    that; the bodies would sit in context for the rest of the run for nothing.
    The file is merged by name in Python, so you never need to reproduce it."""
    path = _pytest_path()
    if not path.exists():
        return {"file_path": str(path), "exists": False,
                "test_names": [], "line_count": 0}
    content = path.read_text(encoding="utf-8")
    try:
        names = sorted(n.name for n in ast.parse(content).body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and n.name.startswith("test_"))
    except SyntaxError:
        names = []
    return {"file_path": str(path), "exists": True, "test_names": names,
            "line_count": len(content.splitlines())}


def run_pytest_tool(context: ToolContext) -> dict:
    """Run the generated test suite on Databricks and capture the result.

    Uploads a driver notebook carrying both pyspark_pytest.py and the converted
    PySpark module it imports, runs it on serverless compute, and returns
    pytest's returncode plus a condensed summary of the failing tests.
    """
    path = _pytest_path()
    if not path.exists():
        return {
            "success": False,
            "error": "No pyspark_pytest.py found. Call add_pytest_tests_tool first.",
        }

    returncode = _run_pytest_suite(context.state)
    return {
        "success": returncode == 0,
        "returncode": returncode,
        "stdout": context.state.get("pytest_last_stdout", ""),
        "stderr": context.state.get("pytest_last_stderr", ""),
    }

parity_test_case_validation_agent = Agent(
    name="parity_test_case_validation_agent",
    model=LiteLlm(
        model=f"databricks/{PARITY_MODEL}",
    ),
    instruction="""You are an expert PySpark test engineer. You write pytest-based
    parity test cases for a converted PySpark pipeline. There is ONE non-negotiable
    rule: EVERY SINGLE FUNCTION listed below MUST have its own `test_<function_name>`
    test. Do not skip any function for any reason — a missing test is a failure.

    The converted PySpark module you are testing is importable as:
        from {pyspark_module_name} import <function_name>

    Functions you MUST write a test for (write one `test_<name>` per function):
    <functions_to_test>
    {functions_to_test}
    </functions_to_test>

    Result of the previous validation (read this — `missing_functions` lists the
    functions that STILL have no test; you MUST add a test for every one of them):
    <parity_test_status>
    {parity_test_status}
    </parity_test_status>

    THE CONVERTED MODULE IS ON DISK, NOT IN THIS PROMPT. Read only what the
    current batch needs:
      * **read_converted_index_tool()** — every function's name, parameters and
        return. Cheap; call it first to plan your batches.
      * **read_converted_functions_tool(function_names=[...])** — the real bodies
        of just the functions you are testing this turn. Assert against what the
        code actually does; never guess behaviour from a name.

    Output of the previous local pytest run (if the suite failed because a TEST is
    wrong, fix the test; if it failed because the CONVERTED CODE is wrong, keep the
    correct test as-is — the code will be fixed by another agent):
    <pytest_stdout>
    {pytest_last_stdout}
    </pytest_stdout>
    <pytest_stderr>
    {pytest_last_stderr}
    </pytest_stderr>

    HOW TO WORK (batched — follow exactly):
    1. Work through `functions_to_test.names` in BATCHES of about 8-10 functions.
       For each name in the batch define a pytest function called exactly
       `test_<function_name>`. NEVER attempt the whole suite in one response: your
       output is capped at ~8k tokens, and a cut-off response loses the entire
       tool call. Call **read_converted_functions_tool** for exactly that batch to
       see what the functions really do before writing their tests.
    2. Create a shared module-scoped SparkSession fixture using plain
       `SparkSession.builder.getOrCreate()` — the suite runs on Databricks, so it must
       bind to the session already present there. Do NOT call `.master("local[*]")`
       or otherwise try to start a local Spark. Build small
       in-memory DataFrames with `spark.createDataFrame(...)` for inputs. Assert on
       the real behaviour of each function — schema, row counts, and concrete values
       via `df.collect()` — based on the real body you fetched in step 1. Do not
       invent behaviour.
    3. For functions that are hard to assert directly (e.g. session builders or
       orchestrators), still write a `test_<name>` smoke test that calls the function
       and asserts it runs / returns without error.
    4. Call **add_pytest_tests_tool(tests_code=...)** with ONLY that batch. On the
       FIRST call include the imports and the SparkSession fixture as well. The tool
       merges batches by name, so nothing you sent earlier is lost — never resend a
       test that is already in the file. It returns `total_tests_in_file` and
       `test_names` so you can track progress.
    5. Repeat steps 1 and 4, batch after batch, until EVERY name in
       `functions_to_test.names` has a test. Compare `test_names` against that list
       before you finish; a missing test is a failure.
    6. Only once the whole suite is in, run it ONCE with **run_pytest_tool**
       (it executes on Databricks).
    7. CRITICAL RULE: never delete, weaken, or trivialise a test just to make the
       suite pass. If a test correctly reflects the source logic but fails, leave it
       failing — that signals the converted code must be fixed elsewhere.
    8. If `parity_test_status.missing_functions` is non-empty, you did NOT cover
       them — send their `test_<name>` in a further batch.

    Tools:
    - read_converted_index_tool(): signatures of every converted function.
    - read_converted_functions_tool(function_names): real bodies of a batch.
    - add_pytest_tests_tool(tests_code): add a BATCH of 8-10 tests; merges by name.
    - read_pytest_file_tool(): read the current test file.
    - run_pytest_tool(): run the suite on Databricks and get pass/fail + output.
    """,
    tools=[
        read_converted_index_tool,
        read_converted_functions_tool,
        add_pytest_tests_tool,
        read_pytest_file_tool,
        run_pytest_tool,
    ],
    mode="task",
    output_key="test_generation_output",
    before_agent_callback=load_functions_to_test,
    after_agent_callback=check_test_case_status,
)
