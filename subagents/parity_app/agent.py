import os
import ast
from datetime import datetime, timezone
import hashlib
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
from ..conversion_loop.case_fact_validation_agent.tools import run_parser
load_dotenv()

OUTPUTS_DIR = pathlib.Path(__file__).parents[2] / "outputs"
PYTEST_FILENAME = "pyspark_pytest.py"

HOST = os.environ["DATABRICKS_HOST"]
TOKEN = os.environ["DATABRICKS_API_KEY"]
USER = os.environ["USER_ID"]

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}
CONVERTED_SUFFIX = "_spark.py"


def _find_converted_module() -> "pathlib.Path | None":
    """The converted PySpark module to test, without the pipeline's help.

    Standalone runs have no session state to inherit a path from, so the module
    is located on disk. `PARITY_TARGET_FILE` wins when set — that is how you
    point this at a specific file instead of whatever was converted last.

    Otherwise the newest `*_spark.py` in outputs/ is used, because the converter
    writes exactly one such file per source script and the most recent one is
    the run you just finished.
    """
    override = os.environ.get("PARITY_TARGET_FILE")
    if override:
        candidate = pathlib.Path(override)
        return candidate if candidate.is_file() else None
    if not OUTPUTS_DIR.is_dir():
        return None
    found = sorted(
        OUTPUTS_DIR.glob(f"*{CONVERTED_SUFFIX}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return found[0] if found else None


def _pytest_path() -> pathlib.Path:
    return OUTPUTS_DIR / PYTEST_FILENAME

PARITY_RECEIPT = "parity_last_pass.json"

_ALWAYS_RUN_ENV = "PARITY_ALWAYS_RUN"


def _receipt_path() -> pathlib.Path:
    return OUTPUTS_DIR / PARITY_RECEIPT


def _module_digest(script_path) -> str:
    """SHA-256 of the converted module, or "" if unreadable.

    Hashes CONTENT rather than using mtime: a re-run that rewrites the file
    byte-identically should still count as unchanged, and mtime would say it
    changed. An unreadable file returns "" so nothing ever matches and the
    stage runs — failing towards doing the work, never towards skipping it.
    """
    try:
        return hashlib.sha256(
            pathlib.Path(str(script_path)).read_bytes()
        ).hexdigest()
    except OSError:
        return ""


def _already_passed(script_path) -> bool:
    """True when this exact module content already passed a parity run.

    The whole stage is the saving here: writing ~20 tests, several Databricks
    pytest runs and any fix passes, all to re-derive a verdict already reached
    for these exact bytes. Re-running conversion without changing the output —
    a retry, a resumed session, a second pipeline pass — hits this.

    Deliberately conservative. Any doubt (no receipt, unreadable file, digest
    mismatch, malformed JSON) returns False and the stage runs.
    """
    if os.environ.get(_ALWAYS_RUN_ENV) == "1":
        return False
    digest = _module_digest(script_path)
    if not digest:
        return False
    try:
        receipt = json.loads(_receipt_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(receipt, dict)
        and receipt.get("passed") is True
        and receipt.get("sha256") == digest
    )


#: Durable record of the last parity run, pass or fail.
PARITY_RESULT = "parity_result.json"


def _result_path() -> pathlib.Path:
    return OUTPUTS_DIR / PARITY_RESULT


def _write_result(state, status: str, target_names: list, missing: list) -> None:
    """Record the outcome on disk, on EVERY run rather than only a green one.

    The receipt (`parity_last_pass.json`) exists to let an unchanged module skip
    the stage, so it is only written when the suite passes — which left a
    failing run with nothing durable at all. The verdict lived in session state
    and vanished with the session: nothing to attach to a ticket, nothing to
    diff against the previous run, nothing for anyone who was not watching the
    screen.

    Written from the verdict callback, so it always matches the verdict rather
    than whatever the agent last reported. Never raises: a report we could not
    write must not fail a run that otherwise succeeded.
    """
    result = state.get("pytest_last_result") or {}
    covered = [n for n in target_names if n not in set(missing)]
    payload = {
        "status": status,
        "passed": status == "success",
        "module": state.get("converted_pyspark_file_path"),
        "test_file": str(_pytest_path()),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coverage": {
            "functions_total": len(target_names),
            "functions_covered": len(covered),
            "functions_missing": missing,
        },
        "suite": {
            "ran": state.get("pytest_last_returncode") is not None,
            "returncode": state.get("pytest_last_returncode"),
            "failed_count": result.get("failed_count", 0),
            "failed_tests": result.get("failed_tests", []),
            "run_error": result.get("run_error"),
        },
        "message": (state.get("parity_test_status") or {}).get("message", ""),
    }
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        _result_path().write_text(json.dumps(payload, indent=2, default=str),
                                  encoding="utf-8")
    except OSError:
        pass


def _write_receipt(script_path, test_count: int) -> None:
    """Record a green run so an unchanged module can skip next time."""
    digest = _module_digest(script_path)
    if not digest:
        return
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        _receipt_path().write_text(
            json.dumps(
                {
                    "passed": True,
                    "sha256": digest,
                    "module": str(script_path),
                    "test_count": test_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def _skip_agent_response(message: str):
    """Content that makes ADK skip the agent, or None if it cannot.

    Returning `types.Content` from a before-agent callback is ADK's documented
    way to bypass the model call. It is wrapped because the exact import path
    has moved between versions, and a skip optimisation must never be the thing
    that breaks the pipeline: on any failure this returns None, the callback
    falls through, and the stage runs exactly as it did before.
    """
    try:
        from google.genai import types

        return types.Content(role="model", parts=[types.Part(text=message)])
    except Exception:
        return None


def load_functions_to_test(callback_context: CallbackContext):
    """Pre-agent-call callback.

    Reads the converted PySpark script path from state, parses it, and stores
    the list of function names that the agent must generate test cases for
    under state["functions_to_test"]. That is all this callback does — it does
    NOT look at tests or coverage.
    """
    state = callback_context.state
    for key, default in (
        ("pytest_last_stdout", ""),
        ("pytest_last_stderr", ""),
        ("pytest_last_returncode", None),
    ):
        if state.get(key) is None:
            state[key] = default
    script_path = state.get("converted_pyspark_file_path") or _find_converted_module()
    if not script_path:
        message = (
            f"No converted PySpark module found. Looked for '*{CONVERTED_SUFFIX}' "
            f"in {OUTPUTS_DIR}. Run the conversion first, or set "
            f"PARITY_TARGET_FILE to the module you want tested."
        )
        state["functions_to_test"] = {"count": 0, "names": [], "error": message}
        if state.get("parity_test_status") is None:
            state["parity_test_status"] = {
                "status": "false",
                "missing_functions": [],
                "message": message,
            }
        return
    state["converted_pyspark_file_path"] = str(script_path)

    if _already_passed(script_path):
        state["parity_test_status"] = {
            "status": "success",
            "missing_functions": [],
            "message": (
                f"Skipped: '{pathlib.Path(str(script_path)).name}' is byte-for-byte "
                f"identical to the module that last passed parity. Set "
                f"{_ALWAYS_RUN_ENV}=1 to force a full run."
            ),
            "skipped": True,
        }
        state["parity_validation_passed"] = True
        state["functions_to_test"] = {"count": 0, "names": [], "signatures": {}}
        try:
            callback_context.actions.escalate = True
        except Exception:
            pass
        return _skip_agent_response(
            "Parity skipped — the converted module is unchanged since it last passed."
        )

    try:
        parsed = _run_parser_cached(str(script_path))
    except Exception as exc:
        state["functions_to_test"] = {
            "count": 0,
            "names": [],
            "error": f"Failed to parse PySpark script '{script_path}': {exc}",
        }
        if state.get("parity_test_status") is None:
            state["parity_test_status"] = {
                "status": "false",
                "missing_functions": [],
                "message": f"Failed to parse PySpark script '{script_path}': {exc}",
            }
        return

    functions = parsed.get("functions") or []
    function_names = [f.get("name") for f in functions if isinstance(f, dict) and f.get("name")]
    signatures = {}
    for f in functions:
        if not isinstance(f, dict) or not f.get("name"):
            continue
        params = f.get("parameters") or []
        names_only = [
            (pm.get("name") if isinstance(pm, dict) else pm) for pm in params
        ]
        signatures[f["name"]] = [n for n in names_only if n]

    # The FULL list stays in state for the coverage check and the tools; only
    # the current batch is injected into the prompt. Showing 80 names while the
    # agent works on 8 re-sends 72 it cannot act on, every iteration.
    state["functions_to_test"] = {
        "count": len(function_names),
        "names": function_names,
        "signatures": signatures,
    }

    # Which 8 to work on now, decided here rather than left to the model:
    # "the ones with no test yet" is a fact about the file on disk, and reading
    # it from disk each turn also makes the batch self-correcting — anything
    # written last turn drops out without the model having to remember it.
    covered_free = _missing_functions(
        function_names, _extract_test_functions(_pytest_text())
    )
    bodies_all = _batch_bodies(str(script_path), covered_free[:BATCH_SIZE])
    batch: list[str] = []
    used = 0
    for name in covered_free[:BATCH_SIZE]:
        size = len(bodies_all.get(name, ""))
        if batch and used + size > BATCH_CHAR_BUDGET:
            break
        batch.append(name)
        used += size

    # Bodies injected, not fetched. The agent read them with a tool on its first
    # move every single turn, and a tool call is a whole extra model round-trip
    # that re-sends the entire prompt — far more than the bodies are worth.
    #
    # It also makes the turn SELF-CONTAINED: everything needed to write this
    # batch is in the prompt, so nothing has to be remembered from a previous
    # turn and no in-turn tool result has to survive. That is what makes
    # dropping conversation history safe (see PARITY_INCLUDE_CONTENTS).
    remaining = max(0, len(covered_free) - len(batch))
    state["current_batch"] = {
        "names": batch,
        "source": {n: bodies_all[n] for n in batch if n in bodies_all},
        "remaining_after_this_batch": remaining,
        # Coverage is decided HERE, not by the writer. It used to get a
        # get_missing_tests() tool and work out for itself whether anything was
        # left — which is reasoning it does not need to do and got wrong,
        # renaming tests that already counted and re-listing the same functions
        # turns apart. The callback already knows: it computed this batch from
        # the file on disk. So it just says what to do next.
        "is_final_batch": remaining == 0,
        "next_step": (
            "This is the LAST batch. Submit it and STOP — the suite is run for "
            "you as soon as every function has a test, and its result is "
            "handled without you."
            if remaining == 0 else
            f"Submit this batch and STOP. {remaining} function(s) remain and you "
            f"will be re-invoked with the next one."
        ),
    }

    state["pyspark_module_name"] = pathlib.Path(script_path).stem

    if state.get("parity_test_status") is None:
        state["parity_test_status"] = {
            "status": "false",
            "missing_functions": function_names,
            "message": (
                "No test cases generated yet; generate a test_<function_name> "
                "for every function."
            ),
        }

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
    unmatched: list[str] = []

    for test_name in test_functions:
        if not test_name.startswith("test_"):
            continue
        stem = test_name[len("test_"):]
        for fn in by_specificity:
            if stem == fn or stem.startswith(f"{fn}_"):
                covered.add(fn)
                break
        else:
            unmatched.append(stem)

    # Second pass, ignoring a leading underscore on the FUNCTION name. A private
    # helper `_parse_catman_sheet` is naturally tested as `test_parse_catman_sheet`
    # -- writing `test__parse_catman_sheet` looks like a typo, so nobody does it.
    # Exact matching left every such helper permanently uncovered, which meant
    # all_present never became true, the suite never ran, and the helper came
    # back in the next batch forever: the writer rewrote the same tests until
    # the loop ran out of iterations.
    #
    # Deliberately a SECOND pass, not a looser first one: where both `foo` and
    # `_foo` exist, the exact match claims its test before the relaxed rule can
    # take it, and a target already covered is never claimed twice.
    for stem in unmatched:
        for fn in by_specificity:
            if fn in covered:
                continue
            bare = fn.lstrip("_")
            if bare and (stem == bare or stem.startswith(f"{bare}_")):
                covered.add(fn)
                break

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


#: Packages the CONVERTED code may need that a Databricks serverless runtime
#: does not ship. Installed into the pytest driver before the module is
#: imported.
#:
#: openpyxl is here because our own conversion rules put it there: hard rule 5
#: forbids xlwings (it drives desktop Excel and can never run on a cluster) and
#: directs Excel I/O to pandas + openpyxl. pandas cannot read or write .xlsx
#: without it, so the converted module imports it and the parity run then died
#: with ModuleNotFoundError: No module named 'openpyxl' — a rule of ours
#: creating a dependency nothing installed.
#:
#: Only ever install what the module actually imports (see _needed_packages):
#: a blanket install costs cluster time on every run for packages the code may
#: never touch.
RUNTIME_PACKAGE_FOR_IMPORT = {
    "openpyxl": "openpyxl",
    "xlrd": "xlrd",
    "xlsxwriter": "xlsxwriter",
    "pyarrow": "pyarrow",
}

#: Extra pip names to force, comma-separated, when the scan is not enough.
EXTRA_PIP_PACKAGES = [
    name.strip()
    for name in os.environ.get("PARITY_PIP_PACKAGES", "").split(",")
    if name.strip()
]


def _needed_packages(module_src: str, test_src: str) -> list[str]:
    """Pip names the converted module or its tests import and may lack.

    Scans BOTH: pandas reaches openpyxl through an engine rather than a visible
    import, so the module may name it only in a `pd.read_excel(...)` call while
    a test names it directly — and either way the import fails at run time.
    """
    found: set[str] = set()
    for src in (module_src, test_src):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    pkg = RUNTIME_PACKAGE_FOR_IMPORT.get(alias.name.split(".")[0])
                    if pkg:
                        found.add(pkg)
            elif isinstance(node, ast.ImportFrom):
                pkg = RUNTIME_PACKAGE_FOR_IMPORT.get((node.module or "").split(".")[0])
                if pkg:
                    found.add(pkg)
    # pandas Excel I/O needs openpyxl even with no direct import of it.
    if "read_excel" in module_src or "to_excel" in module_src or "ExcelWriter" in module_src:
        found.add("openpyxl")
    return sorted(found | set(EXTRA_PIP_PACKAGES))


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

# Installed before the tests import the module. A failure here is reported
# rather than raised: the suite may still run if the package was only needed by
# a path the tests do not reach, and a pip problem should surface as test
# output, not as an opaque driver crash.
_PIP_LOG = ""
if _PIP_PACKAGES:
    import subprocess
    _p = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *_PIP_PACKAGES],
        capture_output=True, text=True,
    )
    if _p.returncode != 0:
        _PIP_LOG = "pip install %s failed: %s" % (_PIP_PACKAGES, (_p.stderr or "")[-500:])

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
if _PIP_LOG:
    _out = _PIP_LOG + "\n" + _out
dbutils.notebook.exit(json.dumps({"returncode": int(_rc), "stdout": _out}))
'''


def _pytest_driver_source(module_name: str, module_src: str, test_src: str) -> str:
    """Build the notebook source that materialises both files and runs pytest.

    Both payloads are embedded base64-encoded so arbitrary quotes, backslashes
    and newlines in the generated code cannot break out of the notebook source.
    """
    module_b64 = base64.b64encode(module_src.encode("utf-8")).decode()
    test_b64 = base64.b64encode(test_src.encode("utf-8")).decode()
    packages = _needed_packages(module_src, test_src)
    header = (
        f"_MODULE_NAME = {module_name!r}\n"
        f"_MODULE_B64 = {module_b64!r}\n"
        f"_TEST_B64 = {test_b64!r}\n"
        f"_PIP_PACKAGES = {packages!r}\n"
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
        # Structured twin of the summary, for the fixer's prompt. The prose form
        # stays for reading the trace; the agent is handed the dict.
        state["pytest_last_result"] = _structured_pytest_result(stdout, "", returncode) 
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

    Escalates (stops the loop) only when BOTH hold:
      * every target function has a test  (full coverage), AND
      * a fresh Databricks pytest run of the CURRENT file exits 0.

    A complete-but-failing suite deliberately does NOT escalate: the code fixer
    runs next in the loop and repairs the converted module against the pytest
    errors, so the next iteration tests genuinely different code. Escalating
    here would end the run on a failure the loop was about to fix.

    The pytest result is (re)computed HERE on the final test file rather than
    trusting whatever the agent last ran — the agent may have run the suite
    before writing all tests, or added tests after its last run, leaving a stale
    returncode in state. Re-running guarantees the verdict matches the file.
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
        _write_receipt(state.get("converted_pyspark_file_path"), len(target_names))
        _write_result(state, "success", target_names, [])
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
        _write_result(state, "incomplete", target_names, missing)
    else:
        state["parity_test_status"] = {
            "status": "failed",
            "missing_functions": [],
            "message": (
                "All functions have tests, but the suite is failing. Fix the "
                "converted PySpark code against the pytest errors below — fix "
                "the code, never weaken a test."
            ),
            "pytest_output": state.get("pytest_last_stdout") or "",
        }
        state["parity_validation_passed"] = False
        _write_result(state, "failed", target_names, [])
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
    _src, tree = _read_and_parse(path)
    if tree is None:
        return {"exists": False, "functions": [], "error": "could not parse"}

    out = []
    for n in tree.body:
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        entry: dict = {"name": n.name}
        if isinstance(n, ast.ClassDef):
            entry["kind"] = "ClassDef" 
        else:
            if n.args.args:
                entry["parameters"] = [a.arg for a in n.args.args]
            if n.returns:
                entry["returns"] = ast.unparse(n.returns)
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
    src, tree = _read_and_parse(path)
    if tree is None:
        return {"exists": False, "functions": {},
                "not_found": list(function_names or []), "error": "could not parse"}

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
                    body[bk] = raw    
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
    targets = list((context.state.get("functions_to_test") or {}).get("names") or [])
    result = {
        "status": "success",
        "tests_added_this_batch": changed,
        "total_tests_in_file": len(total),
    }

    # One test per function is the rule, and the coverage check cannot enforce
    # it: a suffixed name counts, so `test_x`, `test_x_empty` and
    # `test_x_raises` all satisfy `x` and the extras pass unnoticed. Reported
    # here instead, because unrequested extras are pure cost — they are written
    # once and then re-sent in the prompt on every remaining turn.
    return result


#: Whether the test WRITER receives conversation history.
#:
#: "none" is the point of everything above: the batch, its source bodies and
#: the coverage answer are all recomputed from disk each turn, so a turn needs
#: nothing that happened in an earlier one. History is then pure cost — and in
#: this pipeline it is the dominant cost, because all four stages share one
#: session and parity runs last, inheriting parsing, conversion and semantic
#: validation. That is how a single request reached a 202,272-token prompt
#: against a 200,000 ITPM quota.
#:
#: Now the default. The remaining unknown is whether ADK's "none" also strips
#: the CURRENT turn's own tool responses; if it does, a tool the writer must
#: READ FROM stops working. That is why the rules are injected instead of
#: fetched whenever this is "none" (see _instruction_for): read_rules_tool()
#: would otherwise return rules the model never sees, and it would write tests
#: without the counting rule or the Spark fixture rule.
#:
#: Set PARITY_INCLUDE_CONTENTS=default to revert if the writer stops making
#: progress. The fixer keeps history either way — it reads its own tool results
#: mid-turn and has no equivalent injection.
PARITY_INCLUDE_CONTENTS = os.environ.get("PARITY_INCLUDE_CONTENTS", "none")

#: Hard ceiling on functions per iteration.
BATCH_SIZE = 8

#: Ceiling on ONE injected body before it is truncated with a fetch pointer.
#:
#: Measured on the converted module: 10 of 19 bodies exceed 3,000 characters and
#: account for 79% of the total volume, so the long tail is where the cost is.
#:
#: Not replaced by an ID-and-fetch scheme, because a parity test must assert on
#: real behaviour and the agent therefore needs the body of every function it
#: tests — handing out IDs would buy one round-trip per function instead of one
#: injection. Truncating only the outliers keeps the common case free and makes
#: the model pay a round-trip solely for the bodies where it genuinely needs
#: more than the opening.
MAX_INJECTED_BODY_CHARS = 3000

#: Truncation is only safe when the agent can FETCH the rest. With
#: include_contents="none" it cannot: the truncation marker tells it to call
#: read_converted_functions_tool, the call succeeds, and the response is not
#: visible on the next model call — so the agent stops mid-turn with a batch it
#: can neither finish nor abandon. That is the stall seen at event #4.
#:
#: With history off the bodies are therefore injected whole. The cost is
#: bounded by BATCH_CHAR_BUDGET either way, and a prompt of ~8,500 tokens has
#: room the pipeline never did.
_CAN_FETCH = PARITY_INCLUDE_CONTENTS != "none"
_BODY_CAP = MAX_INJECTED_BODY_CHARS if _CAN_FETCH else 10 ** 9

#: Ceiling on the SOURCE characters injected per iteration; the batch is capped
#: by whichever limit binds first.
#:
#: A fixed count is the wrong unit once bodies are injected rather than fetched:
#: the refactored functions range from ~90 to ~7,700 characters, so "8
#: functions" is anywhere between a small prompt and a very large one. Budgeting
#: by size keeps every turn roughly the same cost. The first function is always
#: taken even if it alone exceeds the budget, otherwise an oversized function
#: would never be offered and the loop would spin on it forever.
BATCH_CHAR_BUDGET = 12000

#: Static rules and workflow, fetched on demand instead of ridden in the prompt.
RULES_FILE = "test_generation_rules.md"


#: Cache of parsed files, keyed by (path, size, mtime).
#:
#: A PROCESS cache, not session state, on purpose: state is echoed into every
#: prompt of every agent in the session, so serialising an AST there would
#: re-send tens of kilobytes per turn — the opposite of the saving. Here the
#: parse is reused for free and costs no tokens at all.
#:
#: Keyed by size AND mtime so the fixer editing the module between iterations
#: invalidates it automatically. A rewrite that changes neither would be missed,
#: which cannot happen for real edits and is why this is not hashed: hashing
#: means reading the whole file on every lookup, which is most of what the cache
#: is avoiding.
_PARSE_CACHE: dict = {}

#: Same idea for the expensive PySpark parser (a 1,471-line analysis, not a
#: plain ast.parse), which ran on every before-agent callback.
_RUN_PARSER_CACHE: dict = {}


def _stat_key(path) -> "tuple | None":
    try:
        st = pathlib.Path(str(path)).stat()
    except OSError:
        return None
    return (str(path), st.st_size, st.st_mtime)


def _read_and_parse(path) -> "tuple[str, ast.Module] | tuple[str, None]":
    """Source text and parsed tree for `path`, reusing the last parse.

    Returns `(src, None)` when the file is unreadable or does not parse — the
    callers all degrade rather than raise, and a failed parse is cached too so a
    broken file is not re-parsed on every tool call.
    """
    key = _stat_key(path)
    if key is None:
        return "", None
    hit = _PARSE_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        src = pathlib.Path(str(path)).read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        src, tree = "", None
    _PARSE_CACHE[key] = (src, tree)
    return src, tree


def _run_parser_cached(script_path: str) -> dict:
    """`run_parser` for `script_path`, reused while the file is unchanged."""
    key = _stat_key(script_path)
    if key is None:
        return {}
    hit = _RUN_PARSER_CACHE.get(key)
    if hit is None:
        hit = run_parser(script_path, follow_imports=False)
        _RUN_PARSER_CACHE[key] = hit
    return hit


def _batch_bodies(script_path: str, names: list[str]) -> dict:
    """Real source of just these functions, keyed by name.

    Read from disk each turn rather than carried in state, so a batch always
    reflects the module as it is now — the fixer edits that module between
    iterations, and a remembered body would be stale exactly when it matters.
    """
    if not names:
        return {}
    src, tree = _read_and_parse(script_path)
    if tree is None:
        return {}
    wanted = set(names)
    out: dict = {}
    for n in tree.body:
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if n.name not in wanted:
            continue
        body = ast.get_source_segment(src, n) or ast.unparse(n)
        if len(body) > _BODY_CAP:
            # Cut at a line boundary so the fragment is still readable code,
            # and say plainly that it is a fragment — a body that merely stops
            # mid-statement invites assertions about logic that was cut off.
            head = body[:_BODY_CAP].rsplit("\n", 1)[0]
            body = (
                head
                + f"\n    # ... TRUNCATED ({len(body):,} chars total). This is the "
                  f"opening only.\n    # Call read_converted_functions_tool(['{n.name}']) "
                  f"for the full body\n    # BEFORE asserting on anything below this "
                  f"point."
            )
        out[n.name] = body
    return out


def _pytest_text() -> str:
    """Current contents of the generated test file, or "" if there is none."""
    path = _pytest_path()
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _rules_text() -> str:
    """The rules markdown, or an explanatory string if it is missing."""
    path = pathlib.Path(__file__).with_name(RULES_FILE)
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"(rules file unavailable: {exc})"


def read_rules_tool(context: ToolContext) -> dict:
    """The full test-generation rules and workflow.

    Call this once at the start of a run, or whenever you are unsure how to
    proceed. It holds the counting rule, the Spark fixture requirements, the
    naming rule and the never-weaken-a-test rule.

    Kept out of the prompt deliberately: it is static, so injecting it would
    re-send the same ~1,600 tokens on every iteration of every run.
    """
    return {"rules": _rules_text()}


def _structured_pytest_result(stdout: str, stderr: str, returncode: "int | None") -> dict:
    """Machine-shaped pytest outcome, in place of prose.

    The fixer needs which tests failed and why — not a rendered report. A dict
    says the same thing in a fraction of the tokens, survives being re-sent on
    later turns, and cannot be misread the way a truncated traceback can.
    """
    failed: list[dict] = []
    for line in (stdout or "").splitlines():
        # Tolerate both shapes: raw pytest ("FAILED x::y - err") and the
        # summarised form, which bullets each failure as "  - FAILED ...".
        stripped = line.strip().lstrip("-").strip()
        for marker in ("FAILED ", "ERROR "):
            if not stripped.startswith(marker):
                continue
            body = stripped[len(marker):]
            token, _, detail = body.partition(" - ")
            name = token.split(" ")[0].rsplit("::", 1)[-1].strip()
            if name and not any(f["test"] == name for f in failed):
                failed.append({"test": name, "error": _cap(detail.strip(), 200)})
    result = {
        "passed": returncode == 0,
        "failed_tests": failed,
        "failed_count": len(failed),
    }
    if returncode != 0 and not failed:
        # Nothing parsed as a test failure: the run itself broke (import error,
        # cluster problem). Pass the raw text through rather than reporting an
        # empty failure list, which would read as "nothing wrong".
        result["run_error"] = _cap(stderr or stdout, 600)
    return result


def _instruction_for(include_contents: str, base: str) -> str:
    """The writer's instruction, with the rules inlined when history is off.

    With history retained, read_rules_tool() is the cheaper route: the rules are
    static, so fetching them once beats re-sending ~860 tokens every turn.

    With `include_contents="none"` that inverts. A tool the model must READ FROM
    only works if the response survives to the next model call, and that is the
    exact behaviour "none" may remove — the model would call the tool, receive
    the rules, and write its tests without ever having seen them. Injecting is
    the only form that cannot fail that way.

    The trade is small in the case that matters: with history off the whole
    prompt is a few thousand tokens, so ~860 of rules is noise against the
    ~176,000 of inherited session history it replaces.
    """
    if include_contents != "none":
        return (
            base
            + "\n\n        Call read_rules_tool() before your first batch, and any "
              "time you are\n        unsure: it holds the coverage rule, the Spark "
              "fixture requirements, the\n        naming rule, and the rule about never "
              "weakening a test. Those rules\n        are binding — not having read them "
              "is not an excuse for breaking them.\n\n        - read_rules_tool(): the "
              "workflow and rules in full.\n"
        )
    return (
        base
        + "\n\n        You do NOT see earlier turns. Everything you need is in this "
          "prompt: the\n        batch, the complete source of every function in it, and "
          "the rules below.\n        Nothing can be fetched, and nothing carries over. "
          "These rules are binding.\n\n"
        + "        <rules>\n"
        + "\n".join("        " + line for line in _rules_text().splitlines())
        + "\n        </rules>\n"
    )


def build_parity_agent(name: str = "parity_test_case_validation_agent"):
    """Construct a FRESH parity agent.

    A factory rather than a module-level singleton because the same agent object
    cannot serve two parents: ADK sets `parent_agent` on every entry in a
    `sub_agents` list, so sharing one instance between the standalone App and
    the orchestrator's pipeline would leave it re-parented by whichever imported
    last — and the standalone run would then be driving an agent owned by the
    pipeline. Each caller builds its own.
    """
    return Agent(
        name=name,
        model=LiteLlm(
            model="databricks/databricks-claude-opus-4-7",
        ),
        instruction=_instruction_for(PARITY_INCLUDE_CONTENTS, """You write pytest parity tests for a converted PySpark module.

        The module is importable as: from {pyspark_module_name} import <function_name>

        Work on THIS BATCH ONLY. These are the functions that still have no test;
        the list is recomputed each turn, so anything you wrote last turn is
        already gone from it:
        <current_batch>
        {current_batch}
        </current_batch>

        `current_batch.source` holds the COMPLETE body of every function in the
        batch. Assert against what that code actually does; never guess
        behaviour from a name. There is nothing more to fetch — everything you
        need to write this batch is already in this prompt.

        Write one `test_<function_name>` for each function in the batch, submit
        them with add_pytest_tests_tool, then STOP. Do not work out what is
        left, whether the suite is complete, or whether it passes — the suite is
        run for you once every function has a test, and the result is handled
        without you. Your only job is writing this batch.

        Assume nothing carries over between turns. Everything you need is in
        this prompt, recomputed from what is on disk right now.

        Tools:
        - add_pytest_tests_tool(tests_code): submit the batch; merges by name, so
          never resend a test that is already in the file.
        """),
        # A tool the model must READ FROM is unusable when its response will
        # not be visible on the next model call. add_pytest_tests_tool and
        # run_pytest_tool are fine — they act, and the verdict is recomputed in
        # the callback regardless of what the agent saw. The read tools are
        # withdrawn rather than left as a trap: with history off, calling one
        # ends the turn with nothing gained.
        # run_pytest_tool is NOT here. check_test_case_status runs the suite
        # itself the moment coverage is complete, so a writer that also ran it
        # meant two Databricks runs per iteration for one answer — and with
        # history off the writer could not see its own result anyway, so it
        # reported "did not produce any result" on a run that had worked.
        #
        # The writer's job is to write tests. Running them, judging them and
        # deciding what happens next all belong to the callback, which is the
        # only place that sees the finished file.
        tools=(
            [
                read_converted_index_tool,
                read_converted_functions_tool,
                add_pytest_tests_tool,
                read_rules_tool,
            ]
            if _CAN_FETCH else
            [add_pytest_tests_tool]
        ),
        include_contents=PARITY_INCLUDE_CONTENTS,
        mode="task",
        output_key="test_generation_output",
        before_agent_callback=load_functions_to_test,
        after_agent_callback=check_test_case_status,
    )
