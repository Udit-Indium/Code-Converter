"""Name analysis: what does a statement read, write, mutate and call?

Everything downstream depends on getting this right. If a read is missed the
generated function takes too few parameters and raises `NameError`; if a write
is missed the value never gets returned and the next function receives a stale
one. So this module is deliberately careful about the cases that a naive
`Name`-counting walk gets wrong:

  * **Comprehension scope.** In `[x for x in rows]`, `x` is local to the
    comprehension — it must not look like a variable the block produces. The
    first iterable, though, IS evaluated in the enclosing scope, so `rows` is a
    genuine read.
  * **Lambdas and nested defs.** Their arguments are local; their defaults and
    decorators are evaluated in the enclosing scope.
  * **In-place mutation.** `df['x'] = 1` loads `df` and stores into it. At the
    language level `df` is only read, but a block doing that is producing a
    value, so it is tracked separately as a *mutation*.
  * **Augmented assignment.** `total += 1` both reads and writes `total`.
"""

from __future__ import annotations

import ast
import builtins

from .categories import classify
from .models import Statement
from .sourcemap import SourceMap

#: Names that always resolve without being passed in.
BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins)) | {
    "__name__", "__file__", "__doc__",
}


def _arg_names(args: ast.arguments) -> set[str]:
    """Every parameter name bound by a function or lambda signature."""
    collected = [*args.args, *args.posonlyargs, *args.kwonlyargs]
    names = {arg.arg for arg in collected}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _root_name(node: ast.AST) -> str | None:
    """Root identifier of an attribute/subscript chain.

    `pd.read_csv(...)` -> `pd`; `df['a'].b` -> `df`; a literal -> None.
    """
    current: ast.AST = node
    while True:
        if isinstance(current, ast.Attribute):
            current = current.value
        elif isinstance(current, ast.Subscript):
            current = current.value
        elif isinstance(current, ast.Call):
            current = current.func
        elif isinstance(current, ast.Name):
            return current.id
        else:
            return None


class NameCollector(ast.NodeVisitor):
    """Collects the names a node reads, writes, mutates and calls.

    Nested scopes are tracked with a stack: a `Store` while the stack is
    non-empty binds a name locally and never escapes, which is what keeps
    comprehension and lambda variables out of a block's outputs.
    """

    def __init__(self) -> None:
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        self.mutates: set[str] = set()
        self.calls: set[str] = set()
        self.roots: set[str] = set()
        self.declared_global: set[str] = set()
        #: Names bound by a compound statement's own header — the `i` of
        #: `for i in ...`, the `fh` of `with ... as fh`, the `exc` of
        #: `except E as exc`. The body reads these, but they are supplied by the
        #: statement itself, so they must never become function parameters.
        self.header_bindings: set[str] = set()
        self._scopes: list[set[str]] = []

    # -- scope helpers ------------------------------------------------------

    def _is_local(self, name: str) -> bool:
        return any(name in scope for scope in self._scopes)

    def _bind(self, name: str) -> None:
        """Record a binding, into the innermost scope if there is one."""
        if self._scopes:
            self._scopes[-1].add(name)
        else:
            self.writes.add(name)

    def _read(self, name: str) -> None:
        if not self._is_local(name):
            self.reads.add(name)

    # -- leaves -------------------------------------------------------------

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._read(node.id)
        else:  # Store or Del
            self._bind(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.calls.add(node.attr)
        root = _root_name(node)
        if root:
            self.roots.add(root)
            if isinstance(node.ctx, (ast.Store, ast.Del)) and not self._is_local(root):
                self.mutates.add(root)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        root = _root_name(node)
        if root:
            self.roots.add(root)
            if isinstance(node.ctx, (ast.Store, ast.Del)) and not self._is_local(root):
                self.mutates.add(root)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        self.generic_visit(node)

    # -- statements needing special handling --------------------------------

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # `x += 1` reads x as well as writing it; the generic walk would only
        # see the Store.
        if isinstance(node.target, ast.Name):
            self._read(node.target.id)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._bind(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                # Unknowable: the module could bind anything. Recorded so the
                # caller can warn rather than silently guess wrong.
                self.declared_global.add("*")
                continue
            self._bind(alias.asname or alias.name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._bind(node.name)
            self.header_bindings.add(node.name)
        self.generic_visit(node)

    def _record_header_targets(self, target: ast.AST) -> None:
        """Note every name bound by a loop or `with` target."""
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                self.header_bindings.add(node.id)

    def _visit_loop(self, node: ast.For | ast.AsyncFor) -> None:
        self._record_header_targets(node.target)
        self.generic_visit(node)

    visit_For = _visit_loop
    visit_AsyncFor = _visit_loop

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._record_header_targets(item.optional_vars)
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_Global(self, node: ast.Global) -> None:
        self.declared_global.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.declared_global.update(node.names)

    # -- nested scopes ------------------------------------------------------

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        # Decorators and defaults are evaluated where the function is DEFINED,
        # so they belong to the enclosing scope.
        for decorator in getattr(node, "decorator_list", []):
            self.visit(decorator)
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        # Annotations are evaluated at definition time too (absent
        # `from __future__ import annotations`), so they read enclosing names.
        return_annotation = getattr(node, "returns", None)
        if return_annotation is not None:
            self.visit(return_annotation)
        for arg in (*node.args.args, *node.args.posonlyargs, *node.args.kwonlyargs):
            if arg.annotation is not None:
                self.visit(arg.annotation)

        if not isinstance(node, ast.Lambda):
            self._bind(node.name)

        self._scopes.append(_arg_names(node.args))
        if isinstance(node, ast.Lambda):
            self.visit(node.body)
        else:
            for stmt in node.body:
                self.visit(stmt)
        self._scopes.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function
    visit_Lambda = _visit_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._bind(node.name)
        self._scopes.append(set())
        for stmt in node.body:
            self.visit(stmt)
        self._scopes.pop()

    def _visit_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        generators = node.generators
        # The outermost iterable is evaluated eagerly, in the ENCLOSING scope —
        # `rows` in `[x for x in rows]` is a real read of the surrounding code.
        if generators:
            self.visit(generators[0].iter)

        self._scopes.append(set())
        for position, generator in enumerate(generators):
            self.visit(generator.target)
            if position:  # inner iterables see the comprehension's own names
                self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self._scopes.pop()

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension


def analyse(node: ast.stmt) -> NameCollector:
    """Run the collector over one statement."""
    collector = NameCollector()
    collector.visit(node)
    return collector


def free_names(node: ast.stmt) -> frozenset[str]:
    """Names a def or class body reads from its enclosing scope.

    Used to find variables that a preserved function closes over. Those cannot
    be turned into locals of a generated function without breaking the def, so
    they have to stay module globals.
    """
    collector = analyse(node)
    return frozenset(collector.reads - BUILTIN_NAMES)


def build_statement(node: ast.stmt, source_map: SourceMap, floor: int) -> Statement:
    """Analyse `node` and pair it with its original text."""
    collector = analyse(node)
    calls = frozenset(collector.calls)
    roots = frozenset(collector.roots)
    return Statement(
        node=node,
        source=source_map.statement_source(node, floor),
        category=classify(calls, roots),
        reads=frozenset(collector.reads - BUILTIN_NAMES),
        writes=frozenset(collector.writes),
        mutates=frozenset(collector.mutates - BUILTIN_NAMES),
        calls=calls,
        roots=roots,
        header_bindings=frozenset(collector.header_bindings),
    )


def infer_signature(
    statements: list[Statement],
    later_reads: frozenset[str],
    module_scope: frozenset[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Work out a block's parameters and return values.

    Parameters are names read before the block creates them — requirement 5's
    "read but not created inside the block". The check is sequential, not
    set-based: in::

        df = load()      # creates df
        df = df.dropna() # reads df, but df already exists here

    `df` is not a parameter, because by the time it is read the block has
    already produced it.

    Names resolvable at module level (imports, constants, existing defs) are
    excluded — otherwise every generated function would take `pd` as an
    argument.

    Returns are names the block produces that something later actually reads.
    A value nothing consumes is not returned.

    Args:
        statements: the block's statements, in order.
        later_reads: every name read by later blocks, plus names closed over by
            preserved defs.
        module_scope: names bound at module level.

    Returns:
        `(params, returns)`, each ordered by first appearance for stable output.
    """
    created: set[str] = set()
    params: list[str] = []
    produced: list[str] = []

    for statement in statements:
        for name in sorted(statement.reads | statement.mutates):
            # `header_bindings` covers the loop and `with` targets the statement
            # binds for its own body: `for i in range(10): data.append(i)` reads
            # `i`, but supplies it, so it is not an input to the block.
            if (
                name in created
                or name in module_scope
                or name in statement.header_bindings
            ):
                continue
            if name not in params:
                params.append(name)
        for name in sorted(statement.produces):
            created.add(name)
            if name not in produced:
                produced.append(name)

    returns = tuple(
        name
        for name in produced
        if name in later_reads and name not in module_scope
    )
    return tuple(params), returns
