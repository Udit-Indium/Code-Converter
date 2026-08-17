from __future__ import annotations
import os
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
    """Refactor a flat script into functions, deterministically.

    Accepts a .ipynb and flattens it first, so this can be the pipeline's first
    stage. The refactor itself uses only `ast` and heuristics — no model sees
    any source code. An LLM is consulted once per block, with a short structured
    summary, purely to choose a function name.

    Sets `python_script_path` in state to the refactored file, which is what
    every later stage reads.

    Args:
        context: agent state.
        script_path: path to the .py (or .ipynb) source.
        use_llm_names: name functions with a model.

            Defaults to False because of the rate limit. Naming is one call per
            block, and the calls are small but SEQUENTIAL and fast — a
            69-function notebook fires ~66 requests in under two minutes. The
            token volume is trivial (~7k) but the request rate is the highest
            of any stage, and with only two entitled endpoints there is no
            spare bucket to absorb it.

            Deterministic naming produces `load_sales_df` / `clean_cube_all`
            style names from the same summary, at zero cost. Pass True to get
            model-generated names when quota is not a concern.

    Returns:
        The generated function names, per-block summaries, and any warnings.
    """
    prepared = notebook_to_python(context, script_path)
    if prepared.get("converted") is False and not prepared.get("python_script_path"):
        return {
            "refactored": False,
            "message": prepared.get("message", f"could not prepare '{script_path}'"),
        }

    source_path = Path(prepared["python_script_path"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / f"{source_path.stem}_refactored.py"

    namer = (
        LLMFunctionNamer(model="databricks/databricks-claude-opus-4-7")
        if use_llm_names
        else None
    )
    config = RefactorConfig(
        blocking=BlockingConfig(),
        namer=namer,
        source_name=source_path.name,
    )

    result = refactor_file(source_path, destination, config)

    if not result.ok:
        # A failed refactor is a missed improvement, not a reason to stop: the
        # parser falls back to this flat script on its own, because it looks for
        # `<stem>_refactored.py` on disk and will simply not find one.
        return {
            "refactored": False,
            "python_script_path": str(source_path),
            "message": f"refactor failed, continuing with the flat script: {result.error}",
            "warnings": result.warnings,
        }

    # Nothing is written to state. The file at `destination` IS the handoff —
    # the parser finds it by naming convention. A state key here would be
    # overwritten by the parser's own call to notebook_to_python, silently
    # undoing this stage.
    return {
        "refactored": True,
        "python_script_path": str(destination),
        "functions": [block.name for block in result.blocks],
        "block_count": len(result.blocks),
        "summaries": result.summaries(),
        "warnings": result.warnings,
        "message": (
            f"Refactored '{source_path.name}' into {len(result.blocks)} function(s) "
            f"plus main(), written to '{destination}'. Later stages should use "
            "python_script_path."
        ),
    }


script_refactor_agent = Agent(
    model=LiteLlm(model="databricks/databricks-claude-opus-4-7"),
    name="script_refactor_agent",
    instruction="""
    You restructure a flat Python script into modular functions before it is
    converted to PySpark.

    Tools:
    refactor_script :- deterministically split a flat script into functions using
      AST analysis. Handles .ipynb input by flattening it first. Returns
      `python_script_path` pointing at the refactored file.

    HOW TO WORK:
    1. Call **refactor_script** ONCE with the source path you were given.
    2. Report the generated function names and the block count. Do NOT rewrite,
       review, or second-guess the generated code — the transformation is
       deterministic and already validated by reparsing.
    3. If it reports `refactored: false`, say so plainly and name the reason.
       The pipeline continues with the flat script; that is expected, not an error
       to retry.
    """,
    tools=[refactor_script],
    mode="single_turn",
)
script_refactor_loop_agent = script_refactor_agent
