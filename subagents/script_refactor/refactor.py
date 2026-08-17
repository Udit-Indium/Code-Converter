from __future__ import annotations
import ast
from dataclasses import dataclass, field
from pathlib import Path

from . import codegen
from .analysis import (
    BUILTIN_NAMES,
    build_statement,
    free_names,
    infer_signature,
)
from .blocking import BlockingConfig, split_into_blocks
from .models import Block, ModuleParts, RefactorResult, Statement
from .naming import DeterministicNamer, FunctionNamer, deduplicate
from .sourcemap import SourceMap
from .summary import build_summary

_DEFINITION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_IMPORT_NODES = (ast.Import, ast.ImportFrom)


@dataclass
class RefactorConfig:
    """Everything tunable about a refactor.

    Attributes:
        blocking: how the statement stream is cut into blocks.
        namer: what names the functions. `None` means deterministic naming,
            which needs no network.
        hoist_constants: pull ALL_CAPS literal assignments to module level.
        source_name: name used in the generated header comment.
    """

    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    namer: FunctionNamer | None = None
    hoist_constants: bool = True
    source_name: str = ""


def _is_constant_assignment(node: ast.stmt) -> bool:
    """True for a module-level constant: `MAX_ROWS = 1000`.

    Requires an ALL_CAPS name AND a literal value. The literal check is what
    makes hoisting safe — moving `MAX = 1000` to the top cannot change
    behaviour, whereas moving `MAX = compute()` obviously could.
    """
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return False
    if not (target.id.isupper() and target.id.isidentifier()):
        return False
    try:
        ast.literal_eval(node.value)
    except (ValueError, TypeError, SyntaxError):
        return False
    return True


def partition(tree: ast.Module, source_map: SourceMap, config: RefactorConfig) -> ModuleParts:
    """Split the module body into the four groups the renderer needs.

    Imports, constants and definitions go to module level untouched; everything
    else becomes the sequential body that gets blocked into functions.
    """
    docstring: str | None = None
    body = list(tree.body)
    floor = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            docstring = source_map.segment(body[0])
            floor = body[0].end_lineno or 0
            body = body[1:]

    parts = ModuleParts(docstring=docstring)
    aliases: dict[str, str] = {}

    for node in body:
        statement = build_statement(node, source_map, floor)
        floor = node.end_lineno or floor

        if isinstance(node, _IMPORT_NODES):
            parts.imports.append(statement)
            _record_aliases(node, aliases)
        elif isinstance(node, _DEFINITION_NODES):
            parts.definitions.append(statement)
        elif config.hoist_constants and _is_constant_assignment(node):
            parts.constants.append(statement)
        else:
            parts.body.append(statement)

    parts.import_aliases = aliases
    parts.definition_free_names = frozenset(
        name
        for statement in parts.definitions
        for name in free_names(statement.node)
    )
    return parts


def _record_aliases(node: ast.stmt, aliases: dict[str, str]) -> None:
    """Map each bound name to the module it actually came from."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            aliases[bound] = alias.name.split(".")[0]
    elif isinstance(node, ast.ImportFrom) and node.module:
        for alias in node.names:
            if alias.name != "*":
                aliases[alias.asname or alias.name] = node.module.split(".")[0]


def _collect_warnings(parts: ModuleParts) -> list[str]:
    """Flag the constructs whose behaviour this tool cannot fully guarantee."""
    warnings: list[str] = []
    all_statements = [*parts.imports, *parts.constants, *parts.definitions, *parts.body]

    for statement in parts.imports:
        node = statement.node
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            warnings.append(
                f"line {node.lineno}: `from ... import *` — names it binds are "
                "invisible to the analyser, so inferred parameters may be wrong."
            )

    for statement in all_statements:
        for node in ast.walk(statement.node):
            if isinstance(node, ast.Global):
                warnings.append(
                    f"line {node.lineno}: top-level `global {', '.join(node.names)}` "
                    "— module state is being managed by hand; review the result."
                )
    return warnings


def _definition_ordering_conflict(parts: ModuleParts) -> str:
    """Detect a def whose signature depends on top-level runtime values.

    `def f(x=THRESHOLD)` is fine when `THRESHOLD` is a hoisted constant, but
    `def f(x=len(df))` depends on `df` existing when the `def` executes. Since
    definitions are hoisted above the generated functions, that would now run
    too early — so the refactor is refused rather than producing code that
    raises at import time.
    """
    produced_by_body: set[str] = set()
    for statement in parts.body:
        produced_by_body |= statement.produces

    for statement in parts.definitions:
        node = statement.node
        eager: list[ast.AST] = list(getattr(node, "decorator_list", []))
        args = getattr(node, "args", None)
        if args is not None:
            eager.extend(args.defaults)
            eager.extend(d for d in args.kw_defaults if d is not None)
        eager.extend(getattr(node, "bases", []))

        for expression in eager:
            for inner in ast.walk(expression):
                if isinstance(inner, ast.Name) and inner.id in produced_by_body:
                    return (
                        f"'{getattr(node, 'name', '?')}' at line {node.lineno} has a "
                        f"decorator, default or base depending on '{inner.id}', which "
                        "is computed by top-level code. Hoisting the definition would "
                        "change when that runs, so the refactor was refused."
                    )
    return ""


def _assign_signatures(blocks: list[Block], parts: ModuleParts) -> None:
    """Infer parameters and returns for every block.

    A block's returns are the names it produces that anything later reads —
    later blocks, or a preserved def closing over the name.
    """
    module_scope = parts.module_scope
    for position, block in enumerate(blocks):
        later_reads: set[str] = set(parts.definition_free_names)
        for following in blocks[position + 1 :]:
            later_reads |= following.reads | following.mutates

        block.params, block.returns = infer_signature(
            block.statements, frozenset(later_reads), module_scope
        )


def _assign_names(blocks: list[Block], parts: ModuleParts, namer: FunctionNamer | None) -> list[str]:
    """Summarise and name every block. Returns any namer warnings."""
    warnings: list[str] = []
    fallback = DeterministicNamer()

    for block in blocks:
        block.summary = build_summary(block, parts.import_aliases)

    proposed: list[str] = []
    for block in blocks:
        assert block.summary is not None
        name = ""
        if namer is not None:
            name = namer.name_for(block.summary, block.category)
        if not name:
            name = fallback.name_for(block.summary, block.category)
        proposed.append(name)

    # Never collide with a module-level name, or the generated function would
    # shadow an import or a preserved def.
    reserved = set(parts.module_scope) | BUILTIN_NAMES | {"main"}
    proposed = [f"{name}_step" if name in reserved else name for name in proposed]

    for block, name in zip(blocks, deduplicate(proposed)):
        block.name = name

    failures = getattr(namer, "failures", [])
    if failures:
        warnings.append(
            f"{len(failures)} naming call(s) fell back to deterministic names: "
            f"{failures[0]}"
        )
    return warnings


def refactor_source(source: str, config: RefactorConfig | None = None) -> RefactorResult:
    """Refactor a flat script held in memory.

    Args:
        source: the complete script text.
        config: pipeline configuration; defaults to deterministic naming.

    Returns:
        A `RefactorResult`. Check `ok` before using `code` — a false `ok` means
        the output failed validation and must not be written.
    """
    config = config or RefactorConfig()

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return RefactorResult(
            ok=False,
            code="",
            error=f"input does not parse at line {exc.lineno}: {exc.msg}",
        )

    source_map = SourceMap(source)
    parts = partition(tree, source_map, config)
    warnings = _collect_warnings(parts)

    conflict = _definition_ordering_conflict(parts)
    if conflict:
        return RefactorResult(ok=False, code="", warnings=warnings, error=conflict)

    if not parts.body:
        return RefactorResult(
            ok=False,
            code="",
            warnings=warnings,
            error="no top-level statements to refactor — the script is already "
                  "just imports, constants and definitions.",
        )

    blocks = split_into_blocks(parts.body, config.blocking)
    _assign_signatures(blocks, parts)
    warnings.extend(_assign_names(blocks, parts, config.namer))

    globals_needed = parts.definition_free_names
    code = codegen.render_module(parts, blocks, globals_needed, config.source_name)

    expected = {
        name
        for statement in parts.definitions
        for name in statement.writes
    }
    ok, error = codegen.validate(code, frozenset(expected))
    if not ok:
        return RefactorResult(ok=False, code="", blocks=blocks, warnings=warnings, error=error)

    return RefactorResult(ok=True, code=code, blocks=blocks, warnings=warnings)


def refactor_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    config: RefactorConfig | None = None,
) -> RefactorResult:
    """Refactor a script on disk.

    The output file is written only after validation passes, so a failed
    refactor never leaves a broken file behind.

    Args:
        input_path: the flat .py script to refactor.
        output_path: where to write. Defaults to `<input>_refactored.py`.
        config: pipeline configuration.
    """
    source_file = Path(input_path)
    config = config or RefactorConfig()
    if not config.source_name:
        config.source_name = source_file.name

    try:
        source = source_file.read_text(encoding="utf-8")
    except OSError as exc:
        return RefactorResult(ok=False, code="", error=f"could not read '{source_file}': {exc}")

    result = refactor_source(source, config)
    if not result.ok:
        return result

    destination = Path(output_path) if output_path else source_file.with_name(
        f"{source_file.stem}_refactored.py"
    )
    try:
        destination.write_text(result.code, encoding="utf-8")
    except OSError as exc:
        return RefactorResult(
            ok=False, code="", blocks=result.blocks,
            warnings=result.warnings, error=f"could not write '{destination}': {exc}",
        )

    result.warnings.append(f"written to {destination}")
    return result
