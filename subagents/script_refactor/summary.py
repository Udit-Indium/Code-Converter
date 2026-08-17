from __future__ import annotations
from .categories import rules
from .models import Block, BlockSummary
MAX_OPERATIONS = 8

_NOISE = frozenset({
    "append", "print", "len", "str", "int", "float", "list", "dict", "set",
    "format", "join", "range", "enumerate", "zip", "sorted", "sum", "min",
    "max", "abs", "round", "type", "isinstance", "open", "close",
})


def _significant_operations(block: Block) -> list[str]:
    """Called names that a heuristic recognises, most meaningful first.

    A name matched by a rule is evidence of what the block does; everything
    else is plumbing. Ranking by the matching rule's priority means the
    strongest signal leads — `read_csv` before `astype`.
    """
    ranked: list[tuple[int, str]] = []
    seen: set[str] = set()

    for rule in rules(): 
        for call in sorted(block.calls):
            if call in seen or call in _NOISE:
                continue
            if rule.matches(frozenset({call}), frozenset()):
                ranked.append((rule.priority, call))
                seen.add(call)

    operations = [call for _, call in ranked]
    if len(operations) < MAX_OPERATIONS:
        for call in sorted(block.calls):
            if len(operations) >= MAX_OPERATIONS:
                break
            if call not in seen and call not in _NOISE and not call.startswith("_"):
                operations.append(call)
                seen.add(call)

    return operations[:MAX_OPERATIONS]


def _libraries(block: Block, import_aliases: dict[str, str]) -> list[str]:
    """Real module names behind the aliases the block touches.

    `pd.read_csv(...)` with `import pandas as pd` in scope reports `pandas`,
    not `pd` — the alias is local trivia, the module is the useful fact.
    """
    found = {
        import_aliases[root]
        for root in block.roots
        if root in import_aliases
    }
    return sorted(found)


def build_summary(block: Block, import_aliases: dict[str, str]) -> BlockSummary:
    """Describe `block` in the structured form the namer consumes.

    `block.params` and `block.returns` must already be inferred — the summary
    reports them as `inputs` and `outputs`.

    Args:
        block: the block to describe.
        import_aliases: alias -> module name, from the module's imports.
    """
    modified = sorted(block.produces)
    return BlockSummary(
        operations=_significant_operations(block),
        libraries=_libraries(block, import_aliases),
        modifies=modified,
        inputs=list(block.params),
        outputs=list(block.returns),
    )
