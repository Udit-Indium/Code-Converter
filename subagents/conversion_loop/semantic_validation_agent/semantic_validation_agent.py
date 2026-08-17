import os
import ast
import base64
import json
import math
import pathlib
import subprocess
import sys
import time
import uuid
import requests

from google.adk.agents import Agent, LoopAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from dotenv import load_dotenv

from ..code_converter import semantic_code_fixer_agent
load_dotenv()

OUTPUTS_DIR = pathlib.Path(__file__).parents[3] / "outputs"
DUMMY_DIR = OUTPUTS_DIR / "semantic_dummy"

PYTHON_RUNNER = "semantic_python_runner.py"
PYSPARK_RUNNER = "semantic_pyspark_runner.py"
PYTHON_OUTPUT = "semantic_python_output.json"
PYSPARK_OUTPUT = "semantic_pyspark_output.json"
SOURCE_MODULE = "semantic_source_pipeline"

FLOAT_TOL = 1e-6

HOST = os.environ["DATABRICKS_HOST"]
TOKEN = os.environ["DATABRICKS_API_KEY"]
USER = os.environ["USER_ID"]

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}


def _outputs_path(name: str) -> pathlib.Path:
    return OUTPUTS_DIR / name


def _canonical_scalar(v):
    """Normalise a single value so pandas/Spark representations compare equal."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        # round to the tolerance so tiny float drift compares equal
        return round(v, 9)
    if isinstance(v, bool):
        return bool(v)
    # timestamps / dates come through as strings from JSON; trim a trailing
    # "T00:00:00" and "Z"/millis noise so ISO variants line up
    if isinstance(v, str):
        s = v.strip()
        if s.endswith("Z"):
            s = s[:-1]
        if s.endswith("T00:00:00") or s.endswith(" 00:00:00"):
            s = s[:-9].rstrip("T ").strip()
        return s
    return v


def _canonical_record(rec: dict) -> dict:
    return {str(k): _canonical_scalar(v) for k, v in rec.items()}


def _sort_key(rec: dict) -> str:
    return json.dumps(rec, sort_keys=True, default=str)


def _normalise_records(records) -> list[dict]:
    if not isinstance(records, list):
        return []
    canon = [_canonical_record(r) for r in records if isinstance(r, dict)]
    return sorted(canon, key=_sort_key)


def _floats_close(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=FLOAT_TOL, abs_tol=FLOAT_TOL)
    return a == b


def _semantic_compare(py_records, spark_records) -> dict:
    """Return {match, differences, message} comparing two lists of row dicts,
    order-insensitively and with float tolerance."""
    py = _normalise_records(py_records)
    sp = _normalise_records(spark_records)

    differences: list[str] = []

    py_cols = set().union(*[set(r) for r in py]) if py else set()
    sp_cols = set().union(*[set(r) for r in sp]) if sp else set()
    if py_cols != sp_cols:
        only_py = sorted(py_cols - sp_cols)
        only_sp = sorted(sp_cols - py_cols)
        if only_py:
            differences.append(f"columns only in python output: {only_py}")
        if only_sp:
            differences.append(f"columns only in pyspark output: {only_sp}")

    if len(py) != len(sp):
        differences.append(
            f"row count differs: python={len(py)} vs pyspark={len(sp)}"
        )

    for i, (rp, rs) in enumerate(zip(py, sp)):
        for col in sorted(set(rp) | set(rs)):
            a, b = rp.get(col), rs.get(col)
            if not _floats_close(a, b):
                differences.append(
                    f"row {i} column '{col}': python={a!r} vs pyspark={b!r}"
                )
        if len(differences) >= 50:
            differences.append("... (further differences truncated)")
            break

    match = not differences
    if match:
        message = f"Outputs match: {len(py)} rows, columns {sorted(py_cols)}."
    else:
        message = (
            "PySpark output does not match the Python output. Fix the converted "
            "PySpark code so its output equals the Python baseline."
        )
    return {"match": match, "differences": differences, "message": message}


def _load_records(path: pathlib.Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _run_script(filename: str) -> dict:
    """Run a script under outputs/ locally (cwd=outputs).

    Refuses anything that touches PySpark: Spark work belongs on Databricks, and
    the dispatch in run_python_script_tool is by filename, so a runner saved
    under an unexpected name must not silently fall through to a local run.
    """
    path = _outputs_path(filename)
    if not path.exists():
        return {"success": False, "returncode": None, "stdout": "",
                "stderr": f"script not found: {path}"}

    head = path.read_text(encoding="utf-8", errors="replace")
    if "pyspark" in head or "SparkSession" in head:
        return {
            "success": False, "returncode": None, "stdout": "",
            "stderr": (f"'{filename}' references PySpark but is not "
                       f"{PYSPARK_RUNNER}, so it cannot be run locally. Save the "
                       f"PySpark runner as {PYSPARK_RUNNER} — it is executed on "
                       "Databricks. Only the pandas baseline runner runs locally."),
        }

    try:
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(OUTPUTS_DIR),
        )
    except subprocess.TimeoutExpired as exc:
        return {"success": False, "returncode": None,
                "stdout": exc.stdout or "",
                "stderr": "execution timed out after 600 seconds."}

    return {
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ---------------------------------------------------------------------------
# Databricks execution of the PySpark runner
# ---------------------------------------------------------------------------

# Body of the driver notebook. Receives _MODULE_NAME / _MODULE_B64 /
# _RUNNER_NAME / _RUNNER_B64 / _OUTPUT_NAME / _DUMMY_FILES as literals from
# _pyspark_driver_source().
#
# The runner is executed with runpy under __main__ so a runner written as a
# plain script behaves exactly as it would locally. Its output JSON is read
# back off the remote filesystem and returned through the notebook exit value,
# because the caller compares files that live in the LOCAL outputs/ directory.
_PYSPARK_DRIVER_BODY = '''
import base64, contextlib, io, json, os, runpy, sys, traceback

_DIR = "/tmp/semantic_run"
os.makedirs(_DIR, exist_ok=True)

with open(os.path.join(_DIR, _MODULE_NAME + ".py"), "wb") as _f:
    _f.write(base64.b64decode(_MODULE_B64))
with open(os.path.join(_DIR, _RUNNER_NAME), "wb") as _f:
    _f.write(base64.b64decode(_RUNNER_B64))

for _rel, _b64 in _DUMMY_FILES.items():
    _dest = os.path.join(_DIR, _rel)
    os.makedirs(os.path.dirname(_dest), exist_ok=True)
    with open(_dest, "wb") as _f:
        _f.write(base64.b64decode(_b64))

if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
os.chdir(_DIR)

_out_path = os.path.join(_DIR, _OUTPUT_NAME)
if os.path.exists(_out_path):
    os.remove(_out_path)   # never let a stale file masquerade as this run

_buf = io.StringIO()
_err = ""
try:
    with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
        runpy.run_path(os.path.join(_DIR, _RUNNER_NAME), run_name="__main__")
except Exception:
    _err = traceback.format_exc()

_records = None
if os.path.exists(_out_path):
    with open(_out_path, "r") as _f:
        _records = _f.read()

dbutils.notebook.exit(json.dumps({
    "stdout": _buf.getvalue()[-20000:],
    "error": _err[-8000:],
    "output": _records,
}))
'''


def _pyspark_driver_source(module_path: "str | None" = None) -> "str | None":
    """Build the notebook source that ships the converted module, the PySpark
    runner and any dummy data to Databricks, then runs the runner.

    Every payload is base64-encoded so arbitrary quotes, backslashes and
    newlines in generated code or CSV data cannot break the notebook source.
    Returns None when a required file is missing.
    """
    runner = _outputs_path(PYSPARK_RUNNER)
    if not runner.exists():
        return None

    # Prefer the authoritative path from state; fall back to the naming
    # convention only when the caller has no state to hand.
    module_file = pathlib.Path(module_path) if module_path else None
    if module_file is None or not module_file.is_file():
        module_file = next(iter(sorted(OUTPUTS_DIR.glob("*_spark.py"))), None)
    if module_file is None or not module_file.is_file():
        return None

    dummy: dict[str, str] = {}
    if DUMMY_DIR.is_dir():
        for f in sorted(DUMMY_DIR.rglob("*")):
            if f.is_file():
                rel = f.relative_to(OUTPUTS_DIR).as_posix()
                dummy[rel] = base64.b64encode(f.read_bytes()).decode()

    header = (
        f"_MODULE_NAME = {module_file.stem!r}\n"
        f"_MODULE_B64 = {base64.b64encode(module_file.read_bytes()).decode()!r}\n"
        f"_RUNNER_NAME = {PYSPARK_RUNNER!r}\n"
        f"_RUNNER_B64 = {base64.b64encode(runner.read_bytes()).decode()!r}\n"
        f"_OUTPUT_NAME = {PYSPARK_OUTPUT!r}\n"
        f"_DUMMY_FILES = {dummy!r}\n"
    )
    return header + _PYSPARK_DRIVER_BODY


def _run_pyspark_runner(module_path: "str | None" = None) -> dict:
    """Run semantic_pyspark_runner.py on Databricks serverless.

    Uploads a driver notebook carrying the runner, the converted PySpark module
    it imports and any dummy data, submits it, then writes the JSON the runner
    produced back to the LOCAL outputs/semantic_pyspark_output.json so the
    existing comparison helpers keep working unchanged.

    Returns the same {success, returncode, stdout, stderr} shape as _run_script.
    """
    driver = _pyspark_driver_source(module_path)
    if driver is None:
        return {"success": False, "returncode": None, "stdout": "",
                "stderr": (f"cannot build the Databricks driver: "
                           f"{PYSPARK_RUNNER} or the converted *_spark.py module "
                           f"is missing from {OUTPUTS_DIR}")}

    file_name = f"semantic_{uuid.uuid4().hex}.py"
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
                "run_name": "semantic_validation",
                "tasks": [
                    {
                        "task_key": "semantic",
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
            life_cycle_state = task["state"]["life_cycle_state"]

            if life_cycle_state in ["TERMINATED", "INTERNAL_ERROR", "SKIPPED"]:
                break

            if time.time() - start > timeout:
                return {"success": False, "returncode": None, "stdout": "",
                        "stderr": (f"Databricks run timed out after {timeout} "
                                   f"seconds (run_id {run_id}).")}

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
            return {"success": False, "returncode": 1, "stdout": "",
                    "stderr": ("the PySpark driver notebook returned no result. "
                               f"Databricks error: {output.get('error') or 'unknown'}")}

        payload = json.loads(result)
        records_json = payload.get("output")

        if records_json:
            _outputs_path(PYSPARK_OUTPUT).write_text(records_json, encoding="utf-8")

        err = payload.get("error") or ""
        ok = not err and bool(records_json)
        if not err and not records_json:
            err = (f"the runner completed but produced no {PYSPARK_OUTPUT}; "
                   "make sure it writes that file.")

        return {
            "success": ok,
            "returncode": 0 if ok else 1,
            "stdout": payload.get("stdout") or "",
            "stderr": err,
        }

    except Exception as exc:
        return {"success": False, "returncode": None, "stdout": "",
                "stderr": f"Databricks execution failed: {type(exc).__name__}: {exc}"}

    finally:
        try:
            requests.post(
                f"{HOST}/api/2.0/workspace/delete",
                headers=HEADERS,
                json={"path": workspace_path, "recursive": False},
            )
        except Exception:
            pass


def _summarize_runner_error(res: dict, converted_name: str, max_chars: int = 900) -> str:
    """Condense a crashed runner's traceback into an actionable one-liner:
    which converted function threw, at what line, and the exception message.

    Without this, a crash only surfaces to the fixer as "pyspark output missing"
    — with no clue which function is broken. PySpark also prepends thousands of
    log4j/WARN lines, so we extract just the Python traceback tail."""
    stderr = res.get("stderr") or ""
    lines = stderr.splitlines()
    exc = ""
    for l in reversed(lines):
        s = l.strip()
        if s and (("Error" in s or "Exception" in s) and ":" in s) \
                and not s.startswith(("WARN", "File ")):
            exc = s
            break

    culprits = []
    for l in lines:
        s = l.strip()
        if converted_name and converted_name in s and ", in " in s:
            part = s.split(converted_name, 1)[1]
            culprits.append(part.strip().lstrip('",').strip())

    where = f" in {culprits[-1]}" if culprits else ""
    summary = f"The converted PySpark pipeline CRASHED{where}. Error: {exc}"
    return summary[:max_chars] if len(summary) > max_chars else summary


def _source_index(src: str) -> str:
    """One line per top-level function/class in the source: name, args, returns.

    Enough for the agent to spot the entrypoint (`run_all`, `run_pipeline`, …)
    and see whether loaders take a path, without the whole script in the prompt.
    """
    if not src.strip():
        return "(source script unavailable)"
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return f"(source has a syntax error: {exc})"
    lines = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ", ".join(a.arg for a in n.args.args)
            ret = f" -> {ast.unparse(n.returns)}" if n.returns else ""
            lines.append(f"  def {n.name}({args}){ret}")
        elif isinstance(n, ast.ClassDef):
            lines.append(f"  class {n.name}")
    return "\n".join(lines) or "(no top-level functions)"


def load_semantic_inputs(callback_context: CallbackContext) -> None:
    """Ensure the source Python is on disk (so a runner can import it) and seed
    the state keys the instruction template references.

    The code parser leaves the source on disk at outputs/source_script.py. We
    read it from there and copy it to outputs/<SOURCE_MODULE>.py so the generated
    runner can `import` it — the content is deliberately not carried in state,
    which would put the whole script into every agent's context. The converted
    PySpark module is already a file at state["converted_pyspark_file_path"].
    """
    state = callback_context.state

    DUMMY_DIR.mkdir(parents=True, exist_ok=True)
    state.setdefault("semantic_match", {
        "match": False,
        "differences": [],
        "message": "Semantic validation has not run yet.",
    })

    # The parser writes this canonical copy of whatever it actually parsed — the
    # refactored script when there is one. Read from the fixed path rather than
    # from state: the parser no longer publishes a path there, and the previous
    # `state.get("source_script_path")` silently yielded None, leaving the
    # semantic runner with no source module at all.
    source_path = OUTPUTS_DIR / "source_script.py"
    try:
        py_content = source_path.read_text(encoding="utf-8")
    except OSError:
        py_content = ""
    if py_content:
        module_path = OUTPUTS_DIR / f"{SOURCE_MODULE}.py"
        try:
            # Write only on a real change. This callback runs before EVERY
            # invocation, and an unconditional write bumped the file's mtime
            # each time — which any mtime-based freshness check downstream then
            # reads as "the code changed".
            unchanged = (
                module_path.is_file()
                and module_path.read_text(encoding="utf-8") == py_content
            )
            if not unchanged:
                module_path.write_text(py_content, encoding="utf-8")
            state["semantic_source_module_name"] = SOURCE_MODULE
        except OSError as exc:
            state["semantic_source_module_name"] = None
            state["semantic_setup_error"] = f"could not write source module: {exc}"
    else:
        state["semantic_source_module_name"] = None

    # A compact map of the source (names + signatures) so the agent can pick the
    # entrypoint and judge the input mode without the whole script in its prompt.
    # It pulls whatever bodies it needs with read_python_file_tool.
    # Still published to state because the instruction template interpolates
    # {source_script_path} to tell the agent where to read bodies from. It is a
    # short path string, not the script's content.
    state["source_script_path"] = str(source_path)
    state["source_index"] = _source_index(py_content or "")

    # This agent's own instruction interpolates {pyspark_module_name}, but the
    # only other writer of that key is the PARITY agent's callback — and it
    # returns early, without setting it, whenever no converted file exists. So
    # relying on that agent having run first left this template variable
    # unresolved exactly when the pipeline was already in trouble. Derive it
    # here from the same source, so the key is always present.
    converted = state.get("converted_pyspark_file_path")
    state["pyspark_module_name"] = (
        pathlib.Path(str(converted)).stem if converted else ""
    )

    state["semantic_dummy_dir"] = str(DUMMY_DIR)
    state["semantic_python_output_path"] = str(_outputs_path(PYTHON_OUTPUT))
    state["semantic_pyspark_output_path"] = str(_outputs_path(PYSPARK_OUTPUT))
    state.setdefault("semantic_pyspark_runner_path", str(_outputs_path(PYSPARK_RUNNER)))
    return None


def _pyspark_output_is_stale(converted_path: "str | None" = None) -> bool:
    """True if the PySpark output does not reflect the current code.

    Only TWO things can invalidate the output: the converted PySpark module and
    the runner that drives it. It previously globbed every `*.py` under
    outputs/, which made the check always true — `load_semantic_inputs` rewrites
    `semantic_source_pipeline.py` on every invocation, and the parser rewrites
    `source_script.py`, so some file in that directory was newer than the output
    every single time. The result was a fresh Databricks run on every loop
    iteration, including iterations where nothing about the PySpark code had
    changed.
    """
    out = _outputs_path(PYSPARK_OUTPUT)
    if not out.exists():
        return True
    out_mtime = out.stat().st_mtime

    deps = [_outputs_path(PYSPARK_RUNNER)]
    if converted_path:
        deps.append(pathlib.Path(str(converted_path)))

    for dep in deps:
        try:
            if dep.exists() and dep.stat().st_mtime > out_mtime:
                return True
        except OSError:
            return True
    return False


def check_semantic_match(callback_context: CallbackContext) -> None:
    """Escalation criteria for the semantic-validation LoopAgent.

    Ensures the PySpark output reflects the current converted code (re-running
    the runner if stale), compares it to the Python baseline, and escalates when
    they match. The verdict is written to state["semantic_match"].
    """
    state = callback_context.state

    converted_path = state.get("converted_pyspark_file_path")
    run_res = None
    if _outputs_path(PYSPARK_RUNNER).exists() and _pyspark_output_is_stale(converted_path):
        run_res = _run_pyspark_runner(converted_path)

    if run_res is not None and not run_res.get("success"):
        converted_name = pathlib.Path(
            state.get("converted_pyspark_file_path") or "").name
        summary = _summarize_runner_error(run_res, converted_name)
        state["semantic_match"] = {
            "match": False,
            "differences": [summary],
            "message": ("The converted PySpark pipeline crashed while producing "
                        "output. Fix the function named in the error above."),
        }
        state["semantic_validation_passed"] = False
        return None

    py_records = _load_records(_outputs_path(PYTHON_OUTPUT))
    sp_records = _load_records(_outputs_path(PYSPARK_OUTPUT))

    if py_records is None:
        state["semantic_match"] = {
            "match": False, "differences": ["python baseline output missing"],
            "message": "Python baseline output was not produced yet.",
        }
        state["semantic_validation_passed"] = False
        return None
    if sp_records is None:
        state["semantic_match"] = {
            "match": False, "differences": ["pyspark output missing"],
            "message": "PySpark output was not produced yet.",
        }
        state["semantic_validation_passed"] = False
        return None

    verdict = _semantic_compare(py_records, sp_records)
    state["semantic_match"] = verdict

    if verdict["match"]:
        state["semantic_validation_passed"] = True
        callback_context.actions.escalate = True
    else:
        state["semantic_validation_passed"] = False
    return None


def read_python_file_tool(context: ToolContext, path: str) -> dict:
    """Read a Python file from disk and return its content.

    The one tool here that legitimately returns code: this agent has to see the
    original pipeline to build a matching dummy dataset and runner. Ask for the
    narrowest file that answers the question — whatever comes back stays in
    context for the rest of the turn."""
    try:
        content = pathlib.Path(path).read_text(encoding="utf-8")
        return {"status": "success", "path": path, "content": content}
    except OSError as exc:
        return {"status": "error", "path": path, "error": str(exc)}


def write_output_file_tool(context: ToolContext, filename: str, file_content: str) -> dict:
    """Write a helper artefact (dummy data file or runner script) under outputs/.

    `filename` may include a subdirectory (e.g. "semantic_dummy/transactions.csv").
    Does NOT touch converted_pyspark_file_path — use the converter's write tool for
    the converted module itself.

    Args:
        filename: path under outputs/, subdirectories allowed.
        file_content: the complete text to write (overwrites any existing file).
    """
    path = OUTPUTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(file_content, encoding="utf-8")
    if filename == PYSPARK_RUNNER:
        context.state["semantic_pyspark_runner_path"] = str(path)
    return {"status": "success", "saved_file_path": str(path)}


def run_python_script_tool(context: ToolContext, filename: str) -> dict:
    """Run a python script located under outputs/ and return
    success / returncode / stdout / stderr.

    The PySpark runner is executed on Databricks serverless (it needs a Spark
    runtime); its output JSON is copied back into outputs/ afterwards. Every
    other script — notably the pandas baseline runner — runs locally.
    """
    if filename == PYSPARK_RUNNER:
        return _run_pyspark_runner(context.state.get("converted_pyspark_file_path"))
    return _run_script(filename)


def compare_outputs_tool(context: ToolContext) -> dict:
    """Compare the saved Python and PySpark outputs semantically (order-insensitive,
    float-tolerant) and record the verdict in state["semantic_match"]."""
    py_records = _load_records(_outputs_path(PYTHON_OUTPUT))
    sp_records = _load_records(_outputs_path(PYSPARK_OUTPUT))
    if py_records is None:
        verdict = {"match": False, "differences": ["python baseline output missing"],
                   "message": "Run the Python runner first."}
    elif sp_records is None:
        verdict = {"match": False, "differences": ["pyspark output missing"],
                   "message": "Run the PySpark runner first."}
    else:
        verdict = _semantic_compare(py_records, sp_records)
    context.state["semantic_match"] = verdict
    return verdict

semantic_validation_agent = Agent(
    name="semantic_validation_agent",
    model=LiteLlm(
        model="databricks/databricks-claude-sonnet-4-6",
    ),
    instruction="""You verify that the converted PySpark pipeline is SEMANTICALLY
    equivalent to the source Python pipeline by running the SAME data through both
    and comparing the outputs.

    The source Python module is importable (working directory = outputs/) as:
        from {semantic_source_module_name} import ...
    The converted PySpark module is at:
        {converted_pyspark_file_path}
    (import it by its file stem, also available as {pyspark_module_name}).

    THE SOURCE IS ON DISK, NOT IN THIS PROMPT — at {source_script_path}. Here is
    its shape; call **read_python_file_tool(path)** if you need actual bodies:
    <source_index>
    {source_index}
    </source_index>

    Latest comparison verdict (if a previous run already matched, you are done):
    <semantic_match>
    {semantic_match}
    </semantic_match>

    STEPS (skip any step whose artefact already exists — do NOT regenerate dummy
    data or the Python baseline once they exist; only the PySpark side changes
    between loop iterations):

    1. INPUT MODE — decide from the source script whether the pipeline reads
       EXTERNAL data (a file path / CLI arg / env var, pd.read_csv/read_parquet,
       parameterised loaders) or is SELF-CONTAINED (loader functions return
       hardcoded data and ignore external input).
       - External data: generate a small but meaningful dummy dataset and save it
         under the `semantic_dummy/` folder (use **write_output_file_tool** with a
         `semantic_dummy/<name>` filename). Make it exercise the real logic (e.g.
         timestamps close together for any velocity/window logic, multiple
         categories/currencies, a few nulls).
       - Self-contained: do NOT fabricate data.

    2. RUNNERS — write two thin runner scripts with **write_output_file_tool**:
       - `semantic_python_runner.py`: import the source module, obtain the pipeline
         result (call its entrypoint, e.g. run_pipeline(); inject the dummy data
         only in the external-data case), convert the final result to a list of row
         dicts, and write it as JSON to `semantic_python_output.json`.
       - `semantic_pyspark_runner.py`: same for the converted PySpark module,
         collecting the Spark result to rows (`.toPandas().to_dict("records")` or
         `[r.asDict() for r in df.collect()]`), writing to
         `semantic_pyspark_output.json`.
       Both MUST write CANONICAL JSON: `orient=records`, round floats, and render
       timestamps/dates as ISO strings, so the two files are directly comparable.
       Never compare raw stdout (pandas print vs Spark .show() are not comparable).

    3. Run `semantic_python_runner.py` with **run_python_script_tool** to produce
       the Python baseline (only if `semantic_python_output.json` does not already
       exist).

    4. Run `semantic_pyspark_runner.py` with **run_python_script_tool** to produce
       the PySpark output.

    5. Call **compare_outputs_tool** and report the result. You are done for this
       turn once both runners have executed and the comparison has been recorded.

    STRICT: never edit the source Python, the dummy data, or the Python baseline to
    force a match. If the outputs differ it is the CONVERTED PYSPARK code that is
    wrong — leave the evidence in place; the code fixer will repair it and the loop
    will re-run.

    Tools:
    - read_python_file_tool(path): read a Python file if you need the source from disk.
    - write_output_file_tool(filename, file_content): write dummy data / runner scripts under outputs/.
    - run_python_script_tool(filename): run a script under outputs/ and get output.
    - compare_outputs_tool(): compare the two saved outputs and record the verdict.
    """,
    tools=[
        read_python_file_tool,
        write_output_file_tool,
        run_python_script_tool,
        compare_outputs_tool,
    ],
    mode="task",
    output_key="semantic_validation_output",
    before_agent_callback=load_semantic_inputs,
    after_agent_callback=check_semantic_match,
)


semantic_validation_loop_agent = LoopAgent(
    name="semantic_validation_loop_agent",
    description=(
        "Builds a dummy dataset, runs it through both the source Python pipeline "
        "and the converted PySpark pipeline, and compares the outputs; on mismatch "
        "the code fixer repairs the converted PySpark code and the loop re-runs "
        "until the outputs match or max_iterations is reached."
    ),
    sub_agents=[semantic_validation_agent, semantic_code_fixer_agent],
    max_iterations=3,
)
