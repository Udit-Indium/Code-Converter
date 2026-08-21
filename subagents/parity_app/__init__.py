"""Standalone parity-test app: write a pytest suite, run it, repair what fails.

Deliberately NOT a stage in the conversion pipeline. Every sub_agent of the
orchestrator shares one session, so whatever runs last inherits all the events
before it — running fourth put this agent's prompt at 176,000-202,000 tokens
against a 200,000 ITPM quota, almost entirely from conversion history it never
read. A separate app gets a fresh session and starts near zero.

Run it after the pipeline finishes:

    cd <repo>/subagents && adk web        # then pick "parity_app"

It finds its own target — the newest `*_spark.py` in outputs/, or whatever
PARITY_TARGET_FILE names — so it needs nothing handed to it. (It will still use
`converted_pyspark_file_path` if a session happens to carry one, which is what
lets `build_parity_loop()` be dropped back into a sub_agents list unchanged if
you ever want the stage back.)

A failing suite is repaired, not just reported: write tests -> run -> fix the
converted module -> re-test, stopping when every function has a test and the
suite is green. The fixer edits the converted PySpark only, never the tests —
a correct test that fails is the signal the conversion is wrong.

Results land in three places:

  * `outputs/parity_result.json` — the verdict, written on EVERY run: status,
    coverage counts, and each failing test with its error.
  * `outputs/pyspark_pytest.py` — the generated suite itself.
  * `outputs/parity_last_pass.json` — written only on a green run; it is what
    lets an unchanged module skip the whole thing next time
    (see `_already_passed` in agent.py).

Session state carries the same verdict under `parity_test_status` while the run
is live, but that disappears with the session — the JSON is what survives.
"""

from __future__ import annotations
import os

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

#: How often session history is summarised. Tunable without a code edit,
#: because it is the one lever that shrinks the prompt without changing what an
#: agent can see inside a turn — it trades summarisation calls for prompt size,
#: so a wrong value costs money or latency, never correctness.
#:
#: Lowered from 5 to 3 after a run hit the Databricks ITPM ceiling: a single
#: request carried a 202,272-token prompt against a 200,000 ITPM quota, so no
#: pacing could make it fit. Later stages inherit every earlier stage's events
#: in one shared session, so the prompt grows across the whole pipeline rather
#: than within one agent.
COMPACTION_INTERVAL = int(os.environ.get("ADK_COMPACTION_INTERVAL", "3"))
COMPACTION_OVERLAP = int(os.environ.get("ADK_COMPACTION_OVERLAP", "2"))

events_compaction_config = EventsCompactionConfig(
    compaction_interval=COMPACTION_INTERVAL,
    overlap_size=COMPACTION_OVERLAP,
)
app = App(
    name="parity_app",
    root_agent=root_agent,
    events_compaction_config=events_compaction_config,
)
__all__ = ["app", "root_agent", "build_parity_loop", "build_parity_agent"]
