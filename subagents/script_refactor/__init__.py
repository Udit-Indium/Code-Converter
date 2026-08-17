from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import script_refactor_agent, script_refactor_loop_agent

from .blocking import BlockingConfig
from .categories import CategoryRule, register, reset_registry
from .models import Block, BlockSummary, ModuleParts, RefactorResult, Statement
from .naming import DeterministicNamer, LLMFunctionNamer, FunctionNamer
from .refactor import RefactorConfig, refactor_file, refactor_source

__all__ = [
    "Block",
    "BlockSummary",
    "BlockingConfig",
    "CategoryRule",
    "DeterministicNamer",
    "FunctionNamer",
    "LLMFunctionNamer",
    "ModuleParts",
    "RefactorConfig",
    "RefactorResult",
    "Statement",
    "refactor_file",
    "refactor_source",
    "register",
    "reset_registry",
    "script_refactor_agent",
    "script_refactor_loop_agent",
]


def __getattr__(name: str) -> Any:
    """Import the ADK agents only when they are actually asked for.

    `orchestrator_agent` imports `script_refactor_loop_agent` from this package,
    but the refactoring library itself needs neither `google.adk` nor any LLM
    dependency. Deferring the import via PEP 562 keeps the core importable — and
    its tests runnable — in a bare environment, while the orchestrator's import
    still resolves normally.
    """
    if name in ("script_refactor_agent", "script_refactor_loop_agent"):
        from . import agent

        return getattr(agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
