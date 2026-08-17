"""Deterministic refactoring of a flat script into functions.

A pure library: AST analysis in, generated source out. It contains no agent and
makes no model call, so `code_parser` can invoke `refactor_file` inline rather
than running it as its own pipeline stage, and the whole package imports and
runs in an environment with neither `google.adk` nor any LLM dependency
installed.
"""

from __future__ import annotations

from .blocking import BlockingConfig
from .categories import CategoryRule, register, reset_registry
from .models import Block, BlockSummary, ModuleParts, RefactorResult, Statement
from .naming import DeterministicNamer, FunctionNamer
from .refactor import RefactorConfig, refactor_file, refactor_source

__all__ = [
    "Block",
    "BlockSummary",
    "BlockingConfig",
    "CategoryRule",
    "DeterministicNamer",
    "FunctionNamer",
    "ModuleParts",
    "RefactorConfig",
    "RefactorResult",
    "Statement",
    "refactor_file",
    "refactor_source",
    "register",
    "reset_registry",
]
