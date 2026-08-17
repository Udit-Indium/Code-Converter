from google.adk.agents import SequentialAgent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig

from .subagents.code_parser import code_parser_agent
from .subagents.conversion_loop import (
    conversion_loop_agent,
    semantic_validation_loop_agent,
)


root_agent = SequentialAgent(
    name="orchestrator_agent",
    description=(
        "Orchestrates the Python-to-PySpark conversion: parses the source script "
        "(restructuring a flat script into functions first, then building the AST "
        "inventory), runs the conversion loop that converts and fact-checks until "
        "the case facts match, then runs the semantic-validation loop that compares "
        "Python vs PySpark outputs on a dummy dataset and fixes the converted code "
        "until they match."
    ),
    sub_agents=[
        code_parser_agent,
        conversion_loop_agent,
        semantic_validation_loop_agent,
    ],
)

events_compaction_config = EventsCompactionConfig(
    compaction_interval=5,
    overlap_size=2,
)

app = App(
    name="code_converter",
    root_agent=root_agent,
    events_compaction_config=events_compaction_config,
)
