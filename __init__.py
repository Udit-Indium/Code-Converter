"""Python-to-PySpark conversion pipeline.

The rate-limiter import comes FIRST and the ordering is load-bearing: importing
`orchestrator_agent` transitively constructs every agent, and each one captures
its LiteLLM client at construction time. Patching afterwards would leave those
already-built clients pointing at the unlimited functions.

`subagents/__init__.py` installs the same patch, so this line is belt-and-braces
rather than the only guard — but it keeps the ordering requirement visible at
the entrypoint, where someone reordering imports would otherwise not see it.
"""

from .subagents import litellm_patch  # noqa: F401  (side effect: rate limiting)
from .orchestrator_agent import root_agent  # noqa: E402  (must follow the patch)

__all__ = ["root_agent", "litellm_patch"]
