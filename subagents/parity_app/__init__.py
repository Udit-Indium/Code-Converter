"""Parity-test stage: write a pytest suite, run it, repair what fails.

One definition, two entry points.

  * In the pipeline — `build_parity_loop()` is appended to the orchestrator's
    sub_agents (after SEMS). This is the automatic path.
  * On its own — `app` / `root_agent`, for re-checking a module without
    re-converting it. Discovery is by directory, and this package now lives
    inside subagents/, so the standalone launch is:

        cd <repo>/subagents && adk web        # then pick "parity_app"

Either way the target resolves the same: `converted_pyspark_file_path` from
state when the pipeline set it, else the newest `*_spark.py` in outputs/,
overridable with PARITY_TARGET_FILE.

A failing suite is repaired, not just reported: write tests -> run -> fix the
converted module -> re-test, stopping when every function has a test and the
suite is green. The fixer edits the converted PySpark only, never the tests —
a correct test that fails is the signal the conversion is wrong.

An unchanged module skips the whole stage; see `_already_passed` in agent.py.
"""

from __future__ import annotations
from google.adk.agents import LoopAgent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from .agent import build_parity_agent
# The fixer lives with the converter because it uses the converter's tools
# (replace_functions_tool, execute_pyspark_script_tool, the conventions skill)
# to edit the module in place.
from ..conversion_loop.code_converter import build_code_fixer_agent

MAX_ITERATIONS = 12


def build_parity_loop(name: str = "parity_test_agent") -> LoopAgent:
    """Construct a FRESH parity stage, agent and loop both.

    Built per call rather than shared, for the same reason `build_parity_agent`
    is a factory: ADK sets `parent_agent` on every entry of a `sub_agents` list,
    so one instance used by both the standalone App and the orchestrator would
    end up owned by whichever imported last.
    """
    return LoopAgent(
        name=name,
        description=(
            "Writes and runs a pytest parity suite against the converted "
            "PySpark module and repairs that module against any failures, "
            "looping until every function has a test and the suite passes."
        ),
        sub_agents=[
            build_parity_agent(f"{name}_writer"),
            build_code_fixer_agent(f"{name}_fixer"),
        ],
        max_iterations=MAX_ITERATIONS,
    )
root_agent = build_parity_loop("parity_app")

events_compaction_config = EventsCompactionConfig(
    compaction_interval=5,
    overlap_size=2,
)
app = App(
    name="parity_app",
    root_agent=root_agent,
    events_compaction_config=events_compaction_config,
)
__all__ = ["app", "root_agent", "build_parity_loop", "build_parity_agent"]
