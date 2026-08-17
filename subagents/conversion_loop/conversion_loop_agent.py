from google.adk.agents import LoopAgent

from .code_converter import code_convertor_agent
from .case_fact_validation_agent import case_fact_checker_agent

conversion_loop_agent = LoopAgent(
    name="conversion_loop_agent",
    description=(
        "Converts the Python script to PySpark and fact-checks the result, "
        "looping until the case facts match or max_iterations is reached."
    ),
    sub_agents=[code_convertor_agent, case_fact_checker_agent],
    # Budget = ceil(functions / BATCH_SIZE) + headroom for runtime-error fix
    # passes. BATCH_SIZE is 8 (see code_convertor_agent), and the converter
    # stops its turn after each batch, so one iteration converts AT MOST 8
    # functions — fewer when a batch does not fit in its output limit.
    #
    # The previous comment assumed ~20 per iteration and set this to 8, which
    # capped the loop at 64 functions. A notebook restructured by the refactor
    # stage routinely exceeds that (the current source is 69), and the shortfall
    # is silent: the loop simply exhausts its iterations with functions still
    # unconverted, and the parity stage then tests an incomplete module.
    #
    # 16 covers ~128 functions, or ~9 batches plus fix passes for the current
    # source. If your source is much larger, raise this or BATCH_SIZE.
    max_iterations=16,
)

