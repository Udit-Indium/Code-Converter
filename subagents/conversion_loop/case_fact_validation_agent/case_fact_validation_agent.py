import os
import ast
import json
import pathlib

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

from .tools import run_parser
load_dotenv()

#: Where the code parser leaves the AST inventory. A fixed name, so nothing has
#: to be threaded through state to find it.
AST_INVENTORY = pathlib.Path(__file__).parents[3] / "outputs" / "ast_inventory.json"


def _converted_inventory(script_path) -> dict:
    """Names and constants present in the converted file — no function bodies.

    Everything this agent checks (are all functions present, all classes, do the
    constants match) is answerable from names and values alone. Reading the file
    here and summarising it keeps the prompt at a few hundred tokens instead of
    the whole script.
    """
    if not script_path or not os.path.isfile(str(script_path)):
        return {"exists": False, "functions": [], "classes": [], "constants": {},
                "note": "No converted file yet — this is the first iteration."}
    try:
        tree = ast.parse(pathlib.Path(str(script_path)).read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        return {"exists": False, "functions": [], "classes": [], "constants": {},
                "note": f"Converted file could not be parsed: {exc}"}

    functions, classes, constants = [], [], {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id.isupper():
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                constants[node.targets[0].id] = ast.unparse(node.value)

    return {"exists": True, "function_count": len(functions),
            "functions": sorted(functions), "classes": sorted(classes),
            "constants": constants}


def load_facts(callback_context: CallbackContext) -> None:
    """Build the case facts from the AST-parsed inventory.

    Extracts the count and names of functions and classes, and the name/value
    of every module-level constant, then stores them under state["case_facts"]
    so the fact-check agent can compare them against the converted script.

    The inventory is read from `outputs/ast_inventory.json` rather than from
    state: the parser writes it there precisely so the full parse — tens of
    kilobytes for a large script — does not ride along in every prompt. Only
    the small derived `case_facts` goes into state.
    """
    ast_parsed_content = {}
    try:
        loaded = json.loads(AST_INVENTORY.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            ast_parsed_content = loaded
    except (OSError, ValueError, TypeError):
        # Left empty: the guard below already reports an empty inventory as
        # "the parser has not run yet", which is the actual cause.
        ast_parsed_content = {}

    functions = ast_parsed_content.get("functions") or []
    classes = ast_parsed_content.get("classes") or []
    constants = ast_parsed_content.get("constants") or {}

    function_names = [f.get("name") for f in functions if isinstance(f, dict)]
    class_names = [c.get("name") for c in classes if isinstance(c, dict)]

    case_facts = {
        "functions": {
            "count": len(function_names),
            "names": function_names,
        },
        "classes": {
            "count": len(class_names),
            "names": class_names,
        },
        "constants": dict(constants) if isinstance(constants, dict) else {},
    }
    callback_context.state["case_facts"] = case_facts
    callback_context.state["converted_inventory"] = _converted_inventory(
        callback_context.state.get("converted_pyspark_file_path")
    )
    callback_context.state.setdefault("status", {
        "status": "error",
        "function_missing": list(function_names),
        "classes_missing": list(class_names),
        "constant_value_mismatch": [],
        "message": "Case facts not validated yet against the converted script.",
    })
    return None


def check_fact_status(callback_context: CallbackContext) -> None:
    """Deterministic escalation criteria for the enclosing conversion LoopAgent.

    Instead of trusting the LLM's self-reported verdict (which was marking
    conversions as "success" even when functions were still missing), this
    callback AST-parses the converted PySpark script itself and compares it
    field-by-field against state["case_facts"]:

      * every source function name must be present in the converted script,
      * every source class name must be present, and
      * every source module-level constant must be present with a matching value.

    Escalate (stop the loop) only when ALL THREE hold. Otherwise the loop keeps
    iterating so the converter can add the missing functions / classes / fix the
    constant values. The authoritative verdict is written to state["status"],
    overriding whatever the LLM produced.
    """
    state = callback_context.state

    case_facts = state.get("case_facts") or {}
    print(case_facts)
    source_functions = list((case_facts.get("functions") or {}).get("names") or [])
    source_classes = list((case_facts.get("classes") or {}).get("names") or [])
    source_constants = case_facts.get("constants") or {}
    if not isinstance(source_constants, dict):
        source_constants = {}

    # Defensive: with no source facts there is nothing to validate against.
    # Never let that collapse to a trivial "success" (empty missing-lists) —
    # that would escalate and stop the loop on an unvalidated / unconverted file.
    if not source_functions and not source_classes and not source_constants:
        print("case_facts is empty — cannot validate; NOT escalating.")
        state["status"] = {
            "status": "error",
            "function_missing": [],
            "classes_missing": [],
            "constant_value_mismatch": [],
            "message": (
                "case_facts is empty — ast_parsed_content was not available when "
                "the source facts were built, so the conversion cannot be validated."
            ),
        }
        state["fact_check_passed"] = False
        return None

    script_path = state.get("converted_pyspark_file_path")
    if not script_path or not os.path.isfile(script_path):
        state["status"] = {
            "status": "error",
            "function_missing": list(source_functions),
            "classes_missing": list(source_classes),
            "constant_value_mismatch": list(source_constants.keys()),
            "message": (
                "No converted PySpark script found at "
                f"'converted_pyspark_file_path' ({script_path!r}). "
                "Write the converted script before validation can pass."
            ),
        }
        state["fact_check_passed"] = False
        return None

    try:
        parsed = run_parser(script_path, follow_imports=False)
    except Exception as exc:
        state["status"] = {
            "status": "error",
            "function_missing": list(source_functions),
            "classes_missing": list(source_classes),
            "constant_value_mismatch": list(source_constants.keys()),
            "message": f"Failed to parse converted PySpark script '{script_path}': {exc}",
        }
        state["fact_check_passed"] = False
        return None

    converted_functions = {
        f.get("name")
        for f in (parsed.get("functions") or [])
        if isinstance(f, dict) and f.get("name")
    }
    converted_classes = {
        c.get("name")
        for c in (parsed.get("classes") or [])
        if isinstance(c, dict) and c.get("name")
    }
    converted_constants = parsed.get("constants") or {}
    if not isinstance(converted_constants, dict):
        converted_constants = {}

    missing_functions = [fn for fn in source_functions if fn not in converted_functions]
    missing_classes = [cn for cn in source_classes if cn not in converted_classes]
    constant_mismatch = [
        name
        for name, value in source_constants.items()
        if name not in converted_constants or converted_constants.get(name) != value
    ]
    print(f"Missing functions: {missing_functions}")
    print(f"Missing classes: {missing_classes}")
    print(f"Constant value mismatches: {constant_mismatch}")
    all_match = not missing_functions and not missing_classes and not constant_mismatch

    if all_match:
        state["status"] = {
            "status": "success",
            "function_missing": [],
            "classes_missing": [],
            "constant_value_mismatch": [],
            "message": (
                f"All {len(source_functions)} functions, {len(source_classes)} classes "
                f"are present in the converted script and all {len(source_constants)} "
                f"constant values match."
            ),
        }
        print("Case facts match the converted script. Escalating to stop the loop.")
        state["fact_check_passed"] = True
        callback_context.actions.escalate = True
    else:
        print("Case facts do NOT match the converted script. Continuing the loop.")
        state["status"] = {
            "status": "error",
            "function_missing": missing_functions,
            "classes_missing": missing_classes,
            "constant_value_mismatch": constant_mismatch,
            "message": (
                "The converted script does not yet match the source case facts. "
                "Refer to the AST-parsed source file and add the missing functions "
                "and classes, and correct the mismatched constant values."
            ),
        }
        state["fact_check_passed"] = False

    return None

case_fact_checker_agent = Agent(
    name="case_fact_checker_agent",
    model = LiteLlm(
        model="databricks/databricks-claude-opus-4-7",
    ),
    instruction= """You are an expert case fact checker who checks the code and make sure that the coonverted python code
    has same number of functions, classes and has same constant values as source python script has.
    <case_facts>
    {case_facts}
    </case_facts>

    What is actually in the converted file right now — names and constant values
    only; function bodies are irrelevant to this check and are not shown:
    <converted_inventory>
    {converted_inventory}
    </converted_inventory>

    Compare the two: every function and class name in `case_facts` should appear in
    `converted_inventory.functions` / `.classes`, and every constant should be present
    with the SAME value. Report which are missing or mismatched.
    NOTE: Your response is advisory only. The loop's escalation decision is made
    deterministically by AST-parsing the converted script, so report honestly.
    """,
    mode="single_turn",
    output_key="status",
    before_agent_callback=load_facts,
    after_agent_callback=check_fact_status,
)
