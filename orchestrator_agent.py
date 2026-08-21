import os

from google.adk.agents import SequentialAgent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig

from .subagents.code_parser import code_parser_agent
from .subagents.conversion_loop import (
    conversion_loop_agent,
    semantic_validation_loop_agent,
)
# parity_app is deliberately NOT imported here. It is a standalone app with its
# own root_agent and App — see subagents/parity_app/__init__.py. Adding it back
# as a stage costs more than the stage itself: every sub_agent shares one
# session, so whatever runs last inherits all the events before it, and parity
# running fourth carried a 176,000-202,000 token prompt against a 200,000 ITPM
# quota purely from history it never used.


root_agent = SequentialAgent(
    name="orchestrator_agent",
    description=(
        "Orchestrates the Python-to-PySpark conversion: parses the source script "
        "(restructuring a flat script into functions first, then building the AST "
        "inventory), runs the conversion loop that converts and fact-checks until "
        "the case facts match, runs the semantic-validation loop that compares "
        "Python vs PySpark outputs on a dummy dataset and fixes the converted code "
        "until they match. The pytest parity suite is a separate app, run on "
        "demand against the finished module rather than as a stage here."
    ),
    sub_agents=[
        code_parser_agent,
        conversion_loop_agent,
        semantic_validation_loop_agent,
    ],
)

#: How often session history is summarised. Tunable without a code edit,
#: because it is the one lever that shrinks the prompt without changing what an
#: agent can see inside a turn — it trades summarisation calls for prompt size,
#: so a wrong value costs money or latency, never correctness.
#:
#: Lowered from 5 to 3 after a run hit the Databricks ITPM ceiling: a single
#: request carried a 202,272-token prompt against a 200,000 ITPM quota, so no
#: pacing could make it fit. Stages share one session, so the prompt grows
#: across the whole pipeline rather than within one agent — which is why the
#: parity suite was moved out of it entirely.
COMPACTION_INTERVAL = int(os.environ.get("ADK_COMPACTION_INTERVAL", "3"))
COMPACTION_OVERLAP = int(os.environ.get("ADK_COMPACTION_OVERLAP", "2"))

events_compaction_config = EventsCompactionConfig(
    compaction_interval=COMPACTION_INTERVAL,
    overlap_size=COMPACTION_OVERLAP,
)

app = App(
    name="code_converter",
    root_agent=root_agent,
    events_compaction_config=events_compaction_config,
)
