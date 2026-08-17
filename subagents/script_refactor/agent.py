from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext

from ..code_parser.code_parser_agent import notebook_to_python
from .blocking import BlockingConfig
from .naming import LLMFunctionNamer
from .refactor import RefactorConfig, refactor_file


load_dotenv()


OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"


def refactor_script(
    context: ToolContext,
    script_path: str,
    use_llm_names: bool = False,
) -> dict[str, object]:
    """
    Refactor a Python script/notebook into modular functions.

    The actual refactoring is deterministic and performed using AST analysis.
    No source code is sent to an LLM during the refactoring operation unless
    `use_llm_names=True`.

    The refactored file is written to the outputs directory and its path is
    returned for downstream agents.
    """

    # ---------------------------------------------------------
    # 1. Prepare notebook/script
    # ---------------------------------------------------------
    prepared = notebook_to_python(context, script_path)

    if (
        prepared.get("converted") is False
        and not prepared.get("python_script_path")
    ):
        return {
            "refactored": False,
            "message": prepared.get(
                "message",
                f"Could not prepare '{script_path}'",
            ),
        }

    source_path = Path(prepared["python_script_path"])

    # ---------------------------------------------------------
    # 2. Output path
    # ---------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    destination = OUTPUT_DIR / f"{source_path.stem}_refactored.py"

    # ---------------------------------------------------------
    # 3. Optional LLM naming
    #
    # IMPORTANT:
    # Keep this False during normal conversion.
    # True causes one LLM request per block.
    # ---------------------------------------------------------
    namer = None

    if use_llm_names:
        namer = LLMFunctionNamer(
            model="databricks/databricks-claude-opus-4-7"
        )

    # ---------------------------------------------------------
    # 4. Deterministic AST refactoring
    # ---------------------------------------------------------
    config = RefactorConfig(
        blocking=BlockingConfig(),
        namer=namer,
        source_name=source_path.name,
    )

    result = refactor_file(
        source_path,
        destination,
        config,
    )

    # ---------------------------------------------------------
    # 5. Refactor failed
    # ---------------------------------------------------------
    if not result.ok:
        return {
            "refactored": False,
            "python_script_path": str(source_path),
            "message": (
                "Refactoring failed. "
                "Pipeline will continue with the original Python script."
            ),
            "error": str(result.error),
        }

    # ---------------------------------------------------------
    # 6. IMPORTANT:
    # Return ONLY compact metadata.
    #
    # Do NOT return result.summaries().
    # Those summaries can significantly increase the LLM
    # context when the notebook contains many blocks.
    # ---------------------------------------------------------
    function_names = [
        block.name
        for block in result.blocks
    ]

    warnings = result.warnings or []

    return {
        "refactored": True,
        "python_script_path": str(destination),
        "block_count": len(result.blocks),
        "functions": function_names,
        "warning_count": len(warnings),
        "message": (
            f"Successfully refactored '{source_path.name}' "
            f"into {len(result.blocks)} function(s) plus main(). "
            f"Output: '{destination}'."
        ),
    }


# ============================================================
# REFACTOR AGENT
# ============================================================

script_refactor_agent = Agent(
    model=LiteLlm(
        model="databricks/databricks-claude-opus-4-7"
    ),

    name="script_refactor_agent",

    instruction="""
You are the script refactoring stage of a Python-to-PySpark
conversion pipeline.

Your responsibility is ONLY to execute the deterministic
script refactoring operation.

Tool:
- refactor_script

WORKFLOW:

1. Call `refactor_script` EXACTLY ONCE using the source path
   provided by the user/pipeline.

2. Do NOT read or rewrite the source code yourself.

3. Do NOT call the refactoring tool again if it succeeds.

4. Do NOT ask the model to review the generated code.

5. If the tool returns `refactored=True`, report only:
   - refactoring succeeded
   - output path
   - block count
   - function names

6. If the tool returns `refactored=False`, report the failure
   and continue with the original script.

7. Do not reproduce the source code or generated code
   in your response.

IMPORTANT:
The refactoring operation is deterministic AST-based processing.
The model is an orchestrator for the tool, not the refactoring
engine itself.
""",

    tools=[
        refactor_script,
    ],

    mode="single_turn",
)


script_refactor_loop_agent = script_refactor_agent