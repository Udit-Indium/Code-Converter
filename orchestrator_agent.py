import os

from google.adk.agents import SequentialAgent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig

from .subagents.code_parser import code_parser_agent
from .subagents.conversion_loop import (
    conversion_loop_agent,
    semantic_validation_loop_agent,
)
from .subagents.parity_app import build_parity_loop


root_agent = SequentialAgent(
    name="orchestrator_agent",
    description=(
        "Orchestrates the Python-to-PySpark conversion: parses the source script "
        "(restructuring a flat script into functions first, then building the AST "
        "inventory), runs the conversion loop that converts and fact-checks until "
        "the case facts match, runs the semantic-validation loop that compares "
        "Python vs PySpark outputs on a dummy dataset and fixes the converted code "
        "until they match, then writes and runs a pytest parity suite over the "
        "converted module and reports coverage and pass/fail."
    ),
    sub_agents=[
        code_parser_agent,
        conversion_loop_agent,
        semantic_validation_loop_agent,
        build_parity_loop("parity_test_agent"),
    ],
)

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
    name="code_converter",
    root_agent=root_agent,
    events_compaction_config=events_compaction_config,
)
