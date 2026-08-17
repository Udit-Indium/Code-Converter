from __future__ import annotations

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


# ============================================================
# PATHS
# ============================================================

AST_INVENTORY = (
    pathlib.Path(__file__).parents[3]
    / "outputs"
    / "ast_inventory.json"
)


# Keep only a small number of names in the LLM-visible state.
MAX_NAMES_IN_STATE = 20


# ============================================================
# HELPERS
# ============================================================

def _read_ast_inventory() -> dict:
    """
    Read the parser inventory from disk.

    The complete inventory is intentionally NOT stored in ADK state.
    """

    try:
        content = AST_INVENTORY.read_text(encoding="utf-8")
        data = json.loads(content)

        return data if isinstance(data, dict) else {}

    except (OSError, ValueError, TypeError):
        return {}


def _source_facts() -> dict:
    """
    Build compact source facts.

    Only names and constant values are extracted.
    Function/class bodies are never placed in state.
    """

    inventory = _read_ast_inventory()

    functions = inventory.get("functions") or []
    classes = inventory.get("classes") or []
    constants = inventory.get("constants") or {}

    function_names = [
        f.get("name")
        for f in functions
        if isinstance(f, dict) and f.get("name")
    ]

    class_names = [
        c.get("name")
        for c in classes
        if isinstance(c, dict) and c.get("name")
    ]

    return {
        "function_count": len(function_names),
        "function_names": function_names,
        "class_count": len(class_names),
        "class_names": class_names,
        "constants": (
            dict(constants)
            if isinstance(constants, dict)
            else {}
        ),
    }


def _converted_inventory(script_path) -> dict:
    """
    Read only structural information from the converted file.

    No function bodies are returned.
    """

    if not script_path:
        return {
            "exists": False,
            "function_count": 0,
            "functions": [],
            "class_count": 0,
            "classes": [],
            "constants": {},
        }

    if not os.path.isfile(str(script_path)):
        return {
            "exists": False,
            "function_count": 0,
            "functions": [],
            "class_count": 0,
            "classes": [],
            "constants": {},
        }

    try:
        source = pathlib.Path(str(script_path)).read_text(
            encoding="utf-8"
        )

        tree = ast.parse(source)

    except (OSError, SyntaxError):
        return {
            "exists": False,
            "function_count": 0,
            "functions": [],
            "class_count": 0,
            "classes": [],
            "constants": {},
        }

    functions = []
    classes = []
    constants = {}

    for node in tree.body:

        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            functions.append(node.name)

        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)

        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.isupper()
        ):
            try:
                constants[node.targets[0].id] = ast.literal_eval(
                    node.value
                )

            except (ValueError, SyntaxError):
                constants[node.targets[0].id] = ast.unparse(
                    node.value
                )

    return {
        "exists": True,
        "function_count": len(functions),
        "functions": sorted(functions),
        "class_count": len(classes),
        "classes": sorted(classes),
        "constants": constants,
    }


# ============================================================
# INITIAL CALLBACK
# ============================================================

def load_facts(callback_context: CallbackContext) -> None:
    """
    Load compact source facts.

    IMPORTANT:
    Do NOT store the entire function list in state.
    """

    state = callback_context.state

    facts = _source_facts()

    # --------------------------------------------------------
    # EMPTY SOURCE CHECK
    # --------------------------------------------------------

    if (
        facts["function_count"] == 0
        and facts["class_count"] == 0
        and not facts["constants"]
    ):
        state["case_facts"] = {
            "function_count": 0,
            "class_count": 0,
            "constant_count": 0,
        }

        state["converted_inventory"] = {
            "exists": False,
            "function_count": 0,
            "class_count": 0,
            "constant_count": 0,
        }

        state["fact_check_passed"] = False

        state["status"] = {
            "status": "error",
            "function_missing_count": 0,
            "classes_missing_count": 0,
            "constant_value_mismatch_count": 0,
            "message": (
                "Source AST inventory is empty or unavailable."
            ),
        }

        return None

    # --------------------------------------------------------
    # CONVERTED FILE
    # --------------------------------------------------------

    script_path = state.get(
        "converted_pyspark_file_path"
    )

    converted = _converted_inventory(script_path)

    # --------------------------------------------------------
    # ONLY COMPACT INFORMATION GOES INTO STATE
    # --------------------------------------------------------

    state["case_facts"] = {
        "function_count": facts["function_count"],
        "class_count": facts["class_count"],
        "constant_count": len(facts["constants"]),
    }

    state["converted_inventory"] = {
        "exists": converted["exists"],
        "function_count": converted["function_count"],
        "class_count": converted["class_count"],
        "constant_count": len(converted["constants"]),
    }

    # --------------------------------------------------------
    # Keep complete facts locally for deterministic validation.
    #
    # These are NOT directly injected into the model prompt.
    # --------------------------------------------------------

    state["_case_fact_source"] = facts

    return None


# ============================================================
# DETERMINISTIC VALIDATION
# ============================================================

def check_fact_status(callback_context: CallbackContext) -> None:
    """
    Deterministically compare source and converted structures.

    The LLM is NOT trusted for the pass/fail decision.
    """

    state = callback_context.state

    facts = state.get("_case_fact_source") or {}

    source_functions = facts.get(
        "function_names",
        [],
    )

    source_classes = facts.get(
        "class_names",
        [],
    )

    source_constants = facts.get(
        "constants",
        {},
    )

    # --------------------------------------------------------
    # EMPTY SOURCE CHECK
    # --------------------------------------------------------

    if (
        not source_functions
        and not source_classes
        and not source_constants
    ):
        state["status"] = {
            "status": "error",
            "function_missing_count": 0,
            "classes_missing_count": 0,
            "constant_value_mismatch_count": 0,
            "message": (
                "Source case facts are empty. "
                "Validation cannot proceed."
            ),
        }

        state["fact_check_passed"] = False

        return None

    # --------------------------------------------------------
    # CONVERTED FILE
    # --------------------------------------------------------

    script_path = state.get(
        "converted_pyspark_file_path"
    )

    if (
        not script_path
        or not os.path.isfile(script_path)
    ):
        state["status"] = {
            "status": "error",
            "function_missing_count": len(
                source_functions
            ),
            "classes_missing_count": len(
                source_classes
            ),
            "constant_value_mismatch_count": len(
                source_constants
            ),
            "message": (
                "Converted PySpark file does not exist yet."
            ),
        }

        state["fact_check_passed"] = False

        return None

    # --------------------------------------------------------
    # PARSE CONVERTED FILE
    # --------------------------------------------------------

    try:
        parsed = run_parser(
            script_path,
            follow_imports=False,
        )

    except Exception as exc:

        state["status"] = {
            "status": "error",
            "function_missing_count": len(
                source_functions
            ),
            "classes_missing_count": len(
                source_classes
            ),
            "constant_value_mismatch_count": len(
                source_constants
            ),
            "message": (
                f"Failed to parse converted file: "
                f"{type(exc).__name__}: {exc}"
            ),
        }

        state["fact_check_passed"] = False

        return None

    # --------------------------------------------------------
    # EXTRACT CONVERTED NAMES
    # --------------------------------------------------------

    converted_functions = {
        f.get("name")
        for f in (parsed.get("functions") or [])
        if isinstance(f, dict)
        and f.get("name")
    }

    converted_classes = {
        c.get("name")
        for c in (parsed.get("classes") or [])
        if isinstance(c, dict)
        and c.get("name")
    }

    converted_constants = (
        parsed.get("constants") or {}
    )

    if not isinstance(
        converted_constants,
        dict,
    ):
        converted_constants = {}

    # --------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------

    missing_functions = [
        fn
        for fn in source_functions
        if fn not in converted_functions
    ]

    missing_classes = [
        cn
        for cn in source_classes
        if cn not in converted_classes
    ]

    constant_mismatch = [
        name
        for name, value in source_constants.items()
        if (
            name not in converted_constants
            or converted_constants.get(name) != value
        )
    ]

    all_match = (
        not missing_functions
        and not missing_classes
        and not constant_mismatch
    )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    if all_match:

        state["status"] = {
            "status": "success",
            "function_missing_count": 0,
            "classes_missing_count": 0,
            "constant_value_mismatch_count": 0,
            "message": (
                f"All {len(source_functions)} functions, "
                f"{len(source_classes)} classes and "
                f"{len(source_constants)} constants match."
            ),
        }

        state["fact_check_passed"] = True

        # Stop LoopAgent
        callback_context.actions.escalate = True

        return None

    # --------------------------------------------------------
    # FAILURE / CONTINUE
    # --------------------------------------------------------

    state["status"] = {
        "status": "error",

        # IMPORTANT:
        # Do NOT store the huge lists in state.

        "function_missing_count": len(
            missing_functions
        ),

        "classes_missing_count": len(
            missing_classes
        ),

        "constant_value_mismatch_count": len(
            constant_mismatch
        ),

        # Small samples for debugging only.
        "missing_functions_sample": (
            missing_functions[:MAX_NAMES_IN_STATE]
        ),

        "missing_classes_sample": (
            missing_classes[:MAX_NAMES_IN_STATE]
        ),

        "constant_mismatch_sample": (
            constant_mismatch[:MAX_NAMES_IN_STATE]
        ),

        "message": (
            "Converted script does not yet match "
            "the source case facts. Continue conversion."
        ),
    }

    state["fact_check_passed"] = False

    return None


# ============================================================
# CASE FACT CHECKER AGENT
# ============================================================

case_fact_checker_agent = Agent(
    name="case_fact_checker_agent",

    model=LiteLlm(
        model="databricks/databricks-claude-opus-4-7",
    ),

    instruction="""
You are a case-fact validation agent.

Your responsibility is ONLY to provide a short structural
assessment of the Python-to-PySpark conversion.

Validate only:

1. Number of functions
2. Number of classes
3. Number of module-level constants

IMPORTANT:

The actual validation is performed deterministically by the
Python callback.

Do NOT invent results.

The callback directly compares the source AST and converted AST.

Source facts:

<case_facts>
{case_facts}
</case_facts>

Converted inventory:

<converted_inventory>
{converted_inventory}
</converted_inventory>

Report ONLY a short structural assessment.

Do NOT:

- reproduce source code
- reproduce converted code
- reproduce complete function lists
- reproduce complete class lists
- reproduce constant values
- perform your own lengthy analysis
- repeat the validation rules unnecessarily

The deterministic callback is authoritative for pass/fail.
""",


    include_contents="none",

    mode="single_turn",

    output_key="case_fact_validation_output",

    before_agent_callback=load_facts,

    after_agent_callback=check_fact_status,
)