"""Splitting the flat statement sequence into blocks.

Three triggers start a new block, per requirement 4:

  1. **The operation category changes.** A run of `read_csv` calls followed by a
     run of `fillna` calls is two jobs, so it becomes two functions.
  2. **The block gets long.** Past `max_statements` (~25) a function stops being
     readable no matter how coherent it is.
  3. **The dependency chain ends.** If a statement reads nothing the current
     block produced, it is starting fresh rather than continuing the work — a
     natural seam.

Order matters: the triggers are checked cheapest-first, and any one of them cuts.

`UNCATEGORISED` statements never trigger rule 1. They are the connective tissue
between the interesting calls — a `print`, a bare name, an `if` guard — and
cutting on them would shatter the script into one-statement functions.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import UNCATEGORISED, Block, Statement


@dataclass(frozen=True)
class BlockingConfig:
    """Tunables for the splitter.

    Attributes:
        max_statements: soft ceiling on statements per block (requirement 4's
            "approximately 25").
        split_on_category_change: cut when the operation category changes.
        split_on_chain_end: cut when a statement depends on nothing the current
            block produced.
        min_statements: never cut below this size. Stops a run of alternating
            categories from producing a pile of one-line functions.
        keep_chains_together: let an unbroken dependency chain outrank a change
            of category, producing far fewer, larger functions.

            Off by default, which is worth explaining because the opposite
            sounds right: pandas code flips category almost every line
            (`read_csv` -> `fillna` -> `merge`) while working on one frame
            throughout, so suppressing those cuts seems like it would avoid
            trivial functions. Measured on real input it does the reverse. A
            270-statement notebook goes from 67 functions of ~4 statements
            (max 6 parameters) to 27 functions that hit the 25-statement
            ceiling instead — and cutting at an arbitrary ceiling rather than
            at a category boundary means MORE variables are live at the seam,
            pushing the worst signature from 6 parameters to 10. A 16-statement
            script collapses into a single function, which is not a refactor.

            So it stays off, and the category boundary — a real semantic seam —
            does the cutting. Turn it on when you want coarse-grained output.
    """

    max_statements: int = 25
    split_on_category_change: bool = True
    split_on_chain_end: bool = True
    min_statements: int = 2
    keep_chains_together: bool = False


def _chain_ended(block_produces: frozenset[str], statement: Statement) -> bool:
    """True if `statement` uses nothing the block has produced so far.

    A block that has produced nothing yet cannot have a broken chain — that is
    the opening statement, not a seam.
    """
    if not block_produces:
        return False
    consumed = statement.reads | statement.mutates
    if not consumed:
        # Reads nothing at all (e.g. `print("done")`). Not evidence of a new
        # chain, so keep it with what it follows.
        return False
    return not (consumed & block_produces)


def should_split(
    current: list[Statement],
    statement: Statement,
    config: BlockingConfig,
) -> bool:
    """Decide whether `statement` starts a new block."""
    if not current:
        return False

    if len(current) >= config.max_statements:
        return True

    if len(current) < config.min_statements:
        return False

    produces: set[str] = set()
    for existing in current:
        produces |= existing.produces
    chain_broken = _chain_ended(frozenset(produces), statement)

    if config.split_on_category_change:
        block_category = _dominant_category(current)
        category_changed = (
            statement.category != UNCATEGORISED
            and block_category != UNCATEGORISED
            and statement.category != block_category
        )
        # An unbroken chain means this statement is still working on the same
        # data, whatever the category says — keep it with its predecessors.
        if category_changed and not (config.keep_chains_together and not chain_broken):
            return True

    if config.split_on_chain_end and chain_broken:
        return True

    return False


def _dominant_category(statements: list[Statement]) -> str:
    """The category that best describes a run of statements.

    The first categorised statement wins rather than the most frequent one: a
    block is named for the work it sets out to do, and the trailing
    uncategorised statements that follow are incidental.
    """
    for statement in statements:
        if statement.category != UNCATEGORISED:
            return statement.category
    return UNCATEGORISED


def split_into_blocks(
    statements: list[Statement],
    config: BlockingConfig | None = None,
) -> list[Block]:
    """Group consecutive statements into blocks.

    Order is never changed — blocks partition the sequence, so replaying them in
    order replays the original program.

    Args:
        statements: the top-level statements left after imports, constants and
            definitions are held out.
        config: splitting tunables; defaults are requirement 4's.

    Returns:
        Blocks in original order. Empty if `statements` is empty.
    """
    config = config or BlockingConfig()
    blocks: list[Block] = []
    current: list[Statement] = []

    for statement in statements:
        if should_split(current, statement, config):
            blocks.append(
                Block(
                    index=len(blocks),
                    category=_dominant_category(current),
                    statements=current,
                )
            )
            current = []
        current.append(statement)

    if current:
        blocks.append(
            Block(
                index=len(blocks),
                category=_dominant_category(current),
                statements=current,
            )
        )
    return blocks
