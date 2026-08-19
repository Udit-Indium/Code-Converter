from __future__ import annotations
from google.adk.agents import LoopAgent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from .agent import build_parity_agent
try:
    from ..subagents.conversion_loop.code_converter import build_code_fixer_agent
except ImportError:
    from subagents.conversion_loop.code_converter import build_code_fixer_agent

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
