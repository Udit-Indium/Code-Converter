from __future__ import annotations
import ast
from dataclasses import dataclass, field
from typing import Any

UNCATEGORISED = "general"


@dataclass(frozen=True)
class Statement:
    """A single top-level statement and the facts derived from its AST.

    Attributes:
        node: the AST node itself.
        source: the statement's EXACT original text, including any trailing
            same-line comment and any comment block immediately above it.
            Emitting this verbatim is what preserves both comments and
            behaviour — the alternative, `ast.unparse`, discards comments and
            silently reformats.
        category: heuristic category name (see `categories.py`).
        reads: names loaded by the statement, excluding names bound inside its
            own nested scopes (comprehensions, lambdas, nested defs).
        writes: names the statement BINDS at this level (assignment targets,
            loop targets, imports, `except ... as`, nested def/class names).
        mutates: root names mutated in place — the `df` of `df['x'] = 1` or
            `df.attr = 1`. These are reads at the language level but matter as
            outputs, because a block that mutates a frame is producing a value.
        calls: every called function name and every attribute name appearing in
            an attribute chain. Feeds both categorisation and the summary.
        roots: root identifiers of attribute chains (the `pd` of `pd.read_csv`),
            used to work out which libraries a block touches.
        header_bindings: names the statement's own header binds — the `i` of
            `for i in ...`, the `fh` of `with ... as fh`. The body reads them,
            but the statement supplies them, so they are never parameters.
    """

    node: ast.stmt
    source: str
    category: str
    reads: frozenset[str]
    writes: frozenset[str]
    mutates: frozenset[str]
    calls: frozenset[str]
    roots: frozenset[str]
    header_bindings: frozenset[str] = frozenset()

    @property
    def lineno(self) -> int:
        """Line the statement starts on, ignoring any attached comments."""
        return self.node.lineno

    @property
    def produces(self) -> frozenset[str]:
        """Names this statement makes available to later code."""
        return self.writes | self.mutates


@dataclass
class BlockSummary:
    """The structured description of a block.

    This is deliberately tiny: it is the only thing that ever crosses the
    network to an LLM, and the whole design keeps it that way. No source code,
    no AST, no file paths — just names.
    """

    operations: list[str]
    libraries: list[str]
    modifies: list[str]
    inputs: list[str]
    outputs: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        """Return the plain dict form used in the LLM prompt."""
        return {
            "operations": self.operations,
            "libraries": self.libraries,
            "modifies": self.modifies,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }


@dataclass
class Block:
    """A run of consecutive statements destined to become one function.

    `params` and `returns` are inferred by `analysis.infer_signature` and
    `name` is filled in by the naming stage, so all three are empty until those
    stages run.
    """

    index: int
    category: str
    statements: list[Statement] = field(default_factory=list)
    name: str = ""
    params: tuple[str, ...] = ()
    returns: tuple[str, ...] = ()
    summary: BlockSummary | None = None

    def __len__(self) -> int:
        return len(self.statements)

    def _union(self, attr: str) -> frozenset[str]:
        out: set[str] = set()
        for stmt in self.statements:
            out |= getattr(stmt, attr)
        return frozenset(out)

    @property
    def reads(self) -> frozenset[str]:
        return self._union("reads")

    @property
    def writes(self) -> frozenset[str]:
        return self._union("writes")

    @property
    def mutates(self) -> frozenset[str]:
        return self._union("mutates")

    @property
    def calls(self) -> frozenset[str]:
        return self._union("calls")

    @property
    def roots(self) -> frozenset[str]:
        return self._union("roots")

    @property
    def produces(self) -> frozenset[str]:
        """Names this block makes available to blocks that follow it."""
        return self.writes | self.mutates

    @property
    def source(self) -> str:
        """The block's statements as one chunk of original source text."""
        return "\n".join(stmt.source for stmt in self.statements)


@dataclass
class ModuleParts:
    """Module-level material that is held out of the blocks.

    Imports, constants, and existing defs and classes are emitted at module
    level unchanged. Only `body` is split into blocks.
    """

    docstring: str | None
    imports: list[Statement] = field(default_factory=list)
    constants: list[Statement] = field(default_factory=list)
    definitions: list[Statement] = field(default_factory=list)
    body: list[Statement] = field(default_factory=list)
    #: alias -> real module name, e.g. {"pd": "pandas"}. Used for `libraries`.
    import_aliases: dict[str, str] = field(default_factory=dict)
    #: Free names read by the preserved defs and classes. Anything here that a
    #: block produces has to stay a module global — see `codegen.render_main`.
    definition_free_names: frozenset[str] = frozenset()

    @property
    def module_scope(self) -> frozenset[str]:
        """Every name bound at module level.

        A block reading one of these does NOT need it as a parameter — `pd`,
        `MAX_ROWS` and `helper()` all resolve as globals from inside the
        generated functions.
        """
        names: set[str] = set()
        for group in (self.imports, self.constants, self.definitions):
            for stmt in group:
                names |= stmt.writes
        return frozenset(names)


@dataclass
class RefactorResult:
    """Everything the caller needs to judge a refactor.

    `code` is only ever populated when the rendered module parsed cleanly;
    `ok` false means the output was rejected and nothing should be written.
    """

    ok: bool
    code: str
    blocks: list[Block] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    def summaries(self) -> list[dict[str, Any]]:
        """Per-block report, for logging or for an agent to relay."""
        return [
            {
                "function": block.name,
                "category": block.category,
                "statements": len(block),
                "params": list(block.params),
                "returns": list(block.returns),
                **(block.summary.to_dict() if block.summary else {}),
            }
            for block in self.blocks
        ]
