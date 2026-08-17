from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable

from .models import UNCATEGORISED


@dataclass(frozen=True)
class CategoryRule:
    """One heuristic.

    A statement matches if ANY of the four tests hits:

        attributes  an exact called-function or attribute name
                    (`fillna`, `groupby`, `merge`)
        prefixes    a called name starting with this
                    (`read_` catches `read_csv`, `read_parquet`, ...)
        functions   a bare, non-attribute call (`open`)
        roots       the root of an attribute chain, i.e. a library alias
                    (`plt` in `plt.show()`, `sns` in `sns.histplot()`)

    Attributes:
        name: category name, also used for block-boundary comparisons.
        verb: imperative used to build a fallback name (`load` -> `load_sales_df`).
        priority: higher wins when several rules match one statement. Specific
            rules must outrank general ones — `df.plot()` is visualisation even
            though `df` came from a transformation.
    """

    name: str
    verb: str
    attributes: frozenset[str] = frozenset()
    prefixes: frozenset[str] = frozenset()
    functions: frozenset[str] = frozenset()
    roots: frozenset[str] = frozenset()
    priority: int = 0

    def matches(self, calls: frozenset[str], roots: frozenset[str]) -> bool:
        """True if this rule applies to a statement with these names in it."""
        if self.attributes & calls:
            return True
        if self.roots & roots:
            return True
        if self.functions & calls:
            return True
        return any(
            call.startswith(prefix) for prefix in self.prefixes for call in calls
        )

DEFAULT_RULES: list[CategoryRule] = [
    CategoryRule(
        name="visualization",
        verb="plot",
        attributes=frozenset({
            "plot", "hist", "scatter", "bar", "barh", "boxplot", "pie",
            "show", "savefig", "subplots", "figure", "heatmap", "imshow",
        }),
        roots=frozenset({"plt", "sns", "matplotlib", "seaborn", "pyplot"}),
        priority=90,
    ),
    CategoryRule(
        name="prediction",
        verb="predict",
        attributes=frozenset({
            "predict", "predict_proba", "transform_predict", "score",
        }),
        priority=85,
    ),
    CategoryRule(
        name="model_training",
        verb="train",
        attributes=frozenset({"fit", "fit_transform", "train"}),
        priority=80,
    ),
    CategoryRule(
        name="data_writing",
        verb="write",
        # `write` on its own catches the Spark chain `df.write.parquet(...)`,
        # where the interesting name is an intermediate attribute rather than
        # the called method.
        attributes=frozenset({"write", "save", "to_parquet", "to_sql"}),
        prefixes=frozenset({"to_", "write", "save"}),
        priority=75,
    ),
    CategoryRule(
        name="data_loading",
        verb="load",
        # `read` catches `spark.read.csv(...)`; `read_` catches `pd.read_csv`.
        attributes=frozenset({"read", "load", "load_dataset"}),
        prefixes=frozenset({"read_", "load_"}),
        functions=frozenset({"open"}),
        priority=70,
    ),
    CategoryRule(
        name="merge_join",
        verb="combine",
        attributes=frozenset({"merge", "join", "concat", "append", "union"}),
        priority=60,
    ),
    CategoryRule(
        name="aggregation",
        verb="aggregate",
        attributes=frozenset({
            "groupby", "pivot", "pivot_table", "agg", "aggregate",
            "value_counts", "crosstab", "resample", "rollup",
        }),
        priority=55,
    ),
    CategoryRule(
        name="data_cleaning",
        verb="clean",
        attributes=frozenset({
            "fillna", "dropna", "replace", "drop_duplicates", "drop",
            "na", "isnull", "notnull",
        }),
        priority=50,
    ),
    CategoryRule(
        name="transformation",
        verb="transform",
        attributes=frozenset({
            "assign", "rename", "astype", "apply", "map", "applymap",
            "withColumn", "select", "filter", "where", "sort_values",
            "reset_index", "set_index", "melt", "explode", "cast",
        }),
        priority=40,
    ),
]

#: The live registry. `register` appends here; `classify` reads it.
_RULES: list[CategoryRule] = list(DEFAULT_RULES)


def register(rule: CategoryRule) -> None:
    """Add a heuristic to the registry.

    Rules registered later win ties at equal priority, so a caller can shadow a
    built-in category by registering a rule with the same priority.
    """
    _RULES.append(rule)


def reset_registry(rules: Iterable[CategoryRule] | None = None) -> None:
    """Restore the registry to `rules`, or to the built-in defaults.

    Mainly for tests, which must not leak registrations into each other.
    """
    global _RULES
    _RULES = list(DEFAULT_RULES if rules is None else rules)


def rules() -> tuple[CategoryRule, ...]:
    """Snapshot of the current registry, highest priority first."""
    return tuple(sorted(_RULES, key=lambda r: -r.priority))


def classify(calls: frozenset[str], roots: frozenset[str]) -> str:
    """Return the category name for a statement, or `UNCATEGORISED`.

    Args:
        calls: called function names plus attribute names in the statement.
        roots: root identifiers of the statement's attribute chains.
    """
    best: CategoryRule | None = None
    for rule in _RULES:
        if not rule.matches(calls, roots):
            continue
        # `>=` so a later registration wins an exact tie, which is what makes
        # shadowing a built-in rule possible.
        if best is None or rule.priority >= best.priority:
            best = rule
    return best.name if best else UNCATEGORISED


def verb_for(category: str) -> str:
    """Imperative verb for a category, used to build fallback function names."""
    for rule in _RULES:
        if rule.name == category:
            return rule.verb
    return "process"
