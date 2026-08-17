from google.adk.agents import LoopAgent

from .parity_test_case_validation_agent import parity_test_case_validation_agent
from .code_converter import code_fixer_agent

test_correction_loop_agent = LoopAgent(
    name="test_correction_loop_agent",
    description=(
        "Generates and runs PySpark parity tests for the converted pipeline; on "
        "failure, feeds the errors to the code fixer and re-tests until every "
        "function has a passing test or max_iterations is reached."
    ),
    sub_agents=[parity_test_case_validation_agent, code_fixer_agent],
    max_iterations=3,
)
