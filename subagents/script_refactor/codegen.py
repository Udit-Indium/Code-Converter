"""Rendering the refactored module.

Layout of the output, in order:

    module docstring        the original one if there was one
    imports                 unchanged, hoisted to the top
    constants               module-level ALL_CAPS literals
    existing defs/classes   byte-for-byte unchanged (requirement 3)
    generated functions     one per block, body copied verbatim
    main()                  calls them in the original order
    __main__ guard

The one subtlety worth knowing about is the `global` declaration in `main()`.

Wrapping top-level code in functions changes name resolution. Given::

    df = pd.read_csv("s.csv")
    def describe():
        return df.shape        # closes over the module global `df`

if `df` becomes a local of a generated `load_data()` and then a local of
`main()`, `describe()` breaks with `NameError` — the refactor would have
changed behaviour, violating requirement 11. So any name a preserved def reads
from module scope is declared `global` in `main()`, which puts it back where
the def expects to find it.
"""

from __future__ import annotations

import ast
import textwrap

from .models import Block, ModuleParts

INDENT = "    "


def _indent(text: str) -> str:
    """Indent a block of source by one level, leaving blank lines empty."""
    return textwrap.indent(text, INDENT, predicate=lambda line: line.strip() != "")


def render_docstring(block: Block) -> str:
    """A docstring describing what the block does, from its summary.

    Built from the structured summary rather than the source, so it says what
    the analysis actually established and never drifts into speculation.
    """
    summary = block.summary
    lines = [f'{INDENT}"""Auto-generated from the original top-level code.', ""]
    if summary:
        if summary.operations:
            lines.append(f"{INDENT}Operations: {', '.join(summary.operations)}.")
        if summary.libraries:
            lines.append(f"{INDENT}Libraries: {', '.join(summary.libraries)}.")
        if summary.inputs:
            lines.append(f"{INDENT}Inputs: {', '.join(summary.inputs)}.")
        if summary.outputs:
            lines.append(f"{INDENT}Outputs: {', '.join(summary.outputs)}.")
    lines.append(f'{INDENT}"""')
    return "\n".join(lines)


def render_function(block: Block, globals_needed: frozenset[str] = frozenset()) -> str:
    """Render one generated function.

    The body is the block's ORIGINAL source, indented and otherwise untouched —
    no reformatting, no `ast.unparse` round trip, comments intact.

    Args:
        block: the named block, with its signature already inferred.
        globals_needed: names that preserved defs close over. Any of these the
            block CREATES must be declared `global` here, not merely returned:
            a preserved def called from inside this very function looks the name
            up in module scope at call time, which is before `main()` gets the
            chance to assign the returned value.
    """
    signature = f"def {block.name}({', '.join(block.params)}):"
    body = _indent(block.source)

    parts = [signature, render_docstring(block)]

    # A name cannot be both a parameter and a global — that is a SyntaxError.
    # Params are already bound by the caller, so they need no declaration.
    declare = sorted((block.writes | block.mutates) & globals_needed - set(block.params))
    if declare:
        parts.append(
            f"{INDENT}# Bound at module level because a module-level function or\n"
            f"{INDENT}# class closes over {'them' if len(declare) > 1 else 'it'}."
        )
        parts.append(f"{INDENT}global {', '.join(declare)}")

    parts.append(body)
    if block.returns:
        parts.append(f"{INDENT}return {', '.join(block.returns)}")
    return "\n".join(parts)


def render_main(blocks: list[Block], globals_needed: frozenset[str]) -> str:
    """Render `main()`, calling every generated function in original order.

    Args:
        blocks: the blocks, in order, already named and with signatures.
        globals_needed: names that must remain module globals because a
            preserved def or class closes over them.
    """
    lines = ["def main() -> None:", f'{INDENT}"""Run the pipeline in its original order."""']

    declared = sorted(
        name
        for block in blocks
        for name in block.returns
        if name in globals_needed
    )
    if declared:
        lines.append(
            f"{INDENT}# Declared global so the module-level functions and classes"
        )
        lines.append(f"{INDENT}# that close over these names still resolve them.")
        lines.append(f"{INDENT}global {', '.join(declared)}")

    for block in blocks:
        call = f"{block.name}({', '.join(block.params)})"
        if block.returns:
            targets = ", ".join(block.returns)
            lines.append(f"{INDENT}{targets} = {call}")
        else:
            lines.append(f"{INDENT}{call}")

    if len(lines) == 2:  # nothing but the docstring
        lines.append(f"{INDENT}pass")
    return "\n".join(lines)


def render_module(
    parts: ModuleParts,
    blocks: list[Block],
    globals_needed: frozenset[str],
    source_name: str = "",
) -> str:
    """Assemble the complete refactored module.

    Args:
        parts: module-level material held out of the blocks.
        blocks: named blocks with signatures inferred.
        globals_needed: names preserved defs close over.
        source_name: original filename, mentioned in the header comment.

    Returns:
        The module source. Callers must validate it with `validate` before
        writing it anywhere.
    """
    chunks: list[str] = []

    if parts.docstring is not None:
        chunks.append(parts.docstring)

    origin = f" from {source_name}" if source_name else ""
    chunks.append(
        f"# Refactored{origin} by script_refactor.\n"
        "# Existing functions and classes are unchanged; the original top-level\n"
        "# code was grouped into functions and is replayed in order by main()."
    )

    # Imports and constants are one-liners and read better packed; definitions
    # get the blank line PEP 8 expects between top-level blocks.
    for group, separator in (
        (parts.imports, "\n"),
        (parts.constants, "\n"),
        (parts.definitions, "\n\n"),
    ):
        if group:
            chunks.append(separator.join(stmt.source for stmt in group))

    for block in blocks:
        chunks.append(render_function(block, globals_needed))

    chunks.append(render_main(blocks, globals_needed))
    chunks.append('if __name__ == "__main__":\n    main()')

    return "\n\n\n".join(chunk.strip("\n") for chunk in chunks if chunk.strip()) + "\n"


def validate(code: str, expected_definitions: frozenset[str]) -> tuple[bool, str]:
    """Check the rendered module before it is written anywhere.

    Two checks, both cheap and both catching real damage:

      1. it parses (requirement 12);
      2. every preserved def and class is still present by name — proof that
         requirement 3 held and nothing was dropped during assembly.

    Returns:
        `(ok, error)`. `error` is empty when `ok`.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        line = exc.lineno or 0
        rows = code.splitlines()
        context = rows[line - 1].strip() if 1 <= line <= len(rows) else ""
        return False, f"generated code does not parse at line {line}: {exc.msg} ({context!r})"

    present = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    missing = sorted(expected_definitions - present)
    if missing:
        return False, f"preserved definitions lost during rendering: {', '.join(missing)}"

    return True, ""
