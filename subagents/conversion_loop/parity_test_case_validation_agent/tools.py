from __future__ import annotations
import ast
import json
import re
import sys
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _str_vals(node: Any) -> list[str]:
    """Recursively extract every string constant from an AST node."""
    if isinstance(node, list):
        out: list[str] = []
        for n in node:
            out.extend(_str_vals(n))
        return out
    if node is None:
        return []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out = []
        for elt in node.elts:
            out.extend(_str_vals(elt))
        return out
    if isinstance(node, ast.Dict):
        out = []
        for k in node.keys:
            out.extend(_str_vals(k))
        for v in node.values:
            out.extend(_str_vals(v))
        return out
    return []


def _kw(node: ast.Call, name: str) -> "ast.expr | None":
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _safe_eval(node: Any) -> Any:
    if node is None:
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = node.operand
        if isinstance(inner, ast.Attribute) and inner.attr == "inf":
            return float("-inf")
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id in ("np", "numpy"):
            if node.attr == "inf":
                return float("inf")
            if node.attr == "nan":
                return float("nan")
    if isinstance(node, ast.List):
        return [_safe_eval(e) for e in node.elts]
    try:
        val = ast.literal_eval(node)
        return _sanitize(val)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return None


def _sanitize(val: Any) -> Any:
    """Recursively convert any set -> sorted list so the result is always
    JSON-serializable."""
    if isinstance(val, set):
        return sorted([_sanitize(v) for v in val], key=str)
    if isinstance(val, dict):
        return {k: _sanitize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize(v) for v in val]
    return val


_OP_NAMES = {
    ast.Eq: "==", ast.NotEq: "!=",
    ast.Lt: "<",  ast.LtE:   "<=",
    ast.Gt: ">",  ast.GtE:   ">=",
    ast.In: "in", ast.NotIn: "not in",
    ast.Is: "is", ast.IsNot: "is not",
}
_COL_FUNCS = {"col", "column"}

# functions.* aggregate builders
_AGG_FUNCS = {
    "sum", "avg", "mean", "min", "max", "count", "countDistinct",
    "count_distinct", "approx_count_distinct", "sumDistinct", "sum_distinct",
    "stddev", "stddev_samp", "stddev_pop", "variance", "var_samp", "var_pop",
    "collect_list", "collect_set", "first", "last", "kurtosis", "skewness",
    "corr", "covar_samp", "covar_pop", "grouping", "grouping_id", "median",
    "mode", "product", "any_value", "bit_and", "bit_or", "bit_xor",
    "max_by", "min_by", "percentile", "percentile_approx",
}

# window-only ranking / offset functions
_RANKING_FUNCS = {
    "row_number", "rank", "dense_rank", "percent_rank", "ntile",
    "cume_dist", "lag", "lead", "nth_value",
}

_STRING_OPS = {
    "lower", "upper", "trim", "ltrim", "rtrim", "btrim", "initcap",
    "regexp_replace", "regexp_extract", "regexp_extract_all", "split",
    "concat", "concat_ws", "substring", "substr", "substring_index",
    "length", "char_length", "character_length", "lpad", "rpad", "translate",
    "overlay", "instr", "locate", "format_string", "format_number", "repeat",
    "reverse", "ascii", "base64", "unbase64", "encode", "decode", "soundex",
    "levenshtein", "contains", "startswith", "endswith", "like", "rlike",
    "ilike", "left", "right", "sentences", "split_part",
}
_NULL_OPS = {
    "fillna", "fill", "dropna", "isNull", "isNotNull", "isnull", "notnull",
    "coalesce", "isnan", "nanvl", "ifnull", "nullif", "nvl", "nvl2",
    "na",
}
_TYPE_OPS = {
    "cast", "astype", "to_date", "to_timestamp", "to_number", "try_cast",
    "to_utc_timestamp", "from_utc_timestamp", "date_format",
}
_WINDOW_OPS = {
    "over", "rowsBetween", "rangeBetween", "partitionBy",
}
_SORT_OPS = {"orderBy", "orderby", "sort", "sortWithinPartitions"}
_DEDUP_OPS = {
    "dropDuplicates", "drop_duplicates", "distinct",
    "dropDuplicatesWithinWatermark",
}
_RESHAPE_OPS = {
    "pivot", "explode", "explode_outer", "posexplode", "posexplode_outer",
    "stack", "melt", "unpivot", "transpose", "inline", "inline_outer",
}
_UNION_OPS = {"union", "unionAll", "unionByName"}
_UDF_OPS = {"udf", "pandas_udf", "UDF"}
_APPLY_OPS = {"applyInPandas", "mapInPandas", "apply", "applyInArrow",
              "mapInArrow", "transform"}

# Methods whose positional string args are column names (df.select("a","b")).
_COL_ARG_METHODS = {
    "select", "groupBy", "groupby", "orderBy", "sort",
    "sortWithinPartitions", "drop", "dropDuplicates", "drop_duplicates",
    "rollup", "cube", "partitionBy",
}


# ---------------------------------------------------------------------------
# Column-reference resolution helpers
# ---------------------------------------------------------------------------

def _func_name(node: ast.Call) -> "str | None":
    """Return the simple name of a call: F.col -> 'col', col -> 'col'."""
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return None


def _col_ref(node: Any) -> "str | None":
    """Return the column name a node references, else None.
    Recognises F.col("x") / col("x") / F.column("x"), df["x"], df.x .
    """
    if node is None:
        return None
    if isinstance(node, ast.Call):
        if _func_name(node) in _COL_FUNCS and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                return first.value
        return None
    if isinstance(node, ast.Subscript) and not isinstance(node.value, ast.Constant):
        strs = _str_vals(node.slice)
        return strs[0] if len(strs) == 1 else None
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        # df.amount  — attribute access on a bare name reads a column.
        return node.attr
    return None


def _is_column_expr(node: Any) -> bool:
    """True if the expression tree references any Column (used to decide
    whether .alias("x") names an OUTPUT column vs a DataFrame table alias)."""
    for child in ast.walk(node) if isinstance(node, ast.AST) else []:
        if isinstance(child, ast.Call) and _func_name(child) in (
            _COL_FUNCS | _AGG_FUNCS | _RANKING_FUNCS | _STRING_OPS | {"when", "lit", "expr"}
        ):
            return True
        if isinstance(child, ast.Subscript) and not isinstance(child.value, ast.Constant):
            return True
    return False


def _clean_col(s: str) -> "str | None":
    """Filter out SQL-expression-ish strings that are not plain column names."""
    s = s.strip()
    if not s or s == "*":
        return None
    if any(ch in s for ch in "()+-*/ ") or " as " in s.lower():
        return None
    return s


# ---------------------------------------------------------------------------
# Per-function analyser
# ---------------------------------------------------------------------------

class _FunctionAnalyser(ast.NodeVisitor):
    """Walk one FunctionDef body and collect all PySpark transformation
    metadata into the same fields the pandas parser produces."""

    def __init__(self, src_lines: list[str]):
        self.src_lines = src_lines

        self.cols_read:    set[str] = set()
        self.cols_written: set[str] = set()

        self.joins:        list[dict] = []
        self.aggregations: list[dict] = []
        self.filters:      list[dict] = []
        self.window_funcs: list[dict] = []

        self.string_ops:  set[str] = set()
        self.null_ops:    set[str] = set()
        self.type_ops:    set[str] = set()
        self.sort_ops:    set[str] = set()
        self.dedup_ops:   set[str] = set()
        self.reshape_ops: set[str] = set()
        self.all_methods: set[str] = set()

        self.has_apply        = False   # applyInPandas / mapInPandas / transform
        self.has_merge_asof   = False   # kept for schema parity (asof pattern)
        self.has_rolling      = False   # rangeBetween time window
        self.has_window_op    = False   # any Window / .over usage
        self.has_explode      = False
        self.has_pivot        = False
        self.is_udf_candidate = False   # F.udf / F.pandas_udf
        self.has_groupby      = False

        # track local list assignments for variable-subscript resolution
        self._local_lists: dict[str, list[str]] = {}
        self._depth = 0

    # ── column helpers ───────────────────────────────────────────────────

    def _add_col_read(self, node: Any) -> None:
        for s in _str_vals(node):
            c = _clean_col(s)
            if c:
                self.cols_read.add(c)

    def _add_col_written(self, node: Any) -> None:
        for s in _str_vals(node):
            c = _clean_col(s)
            if c:
                self.cols_written.add(c)

    # ── nested function handling ─────────────────────────────────────────

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # decorator-based UDFs: @udf / @F.udf / @pandas_udf
        for dec in node.decorator_list:
            dname = None
            if isinstance(dec, ast.Call):
                dname = _func_name(dec)
            elif isinstance(dec, ast.Attribute):
                dname = dec.attr
            elif isinstance(dec, ast.Name):
                dname = dec.id
            if dname in _UDF_OPS:
                self.is_udf_candidate = True

        if self._depth == 0:
            self._depth += 1
            self.generic_visit(node)
            self._depth -= 1
        else:
            inner = _FunctionAnalyser(self.src_lines)
            inner._depth = 1
            inner._local_lists = dict(self._local_lists)
            inner.generic_visit(node)
            if inner.is_udf_candidate or inner.has_apply:
                self.is_udf_candidate = self.is_udf_candidate or inner.is_udf_candidate
            self.cols_read.update(inner.cols_read)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    # ── assignment targets ───────────────────────────────────────────────

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    vals = [
                        elt.value for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
                    if vals:
                        self._local_lists[target.id] = vals
            if isinstance(target, ast.Subscript):
                self._add_col_written(target.slice)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Subscript):
            self._add_col_written(node.target.slice)
        if node.value is not None:
            self.visit(node.value)

    _CONFIG_NAMES = {
        "CONFIG", "config", "TIER_THRESHOLDS", "tier_thresholds",
        "thr", "thresholds", "settings", "params", "kwargs", "conf",
    }

    # ── subscript reads: df["col"] / df[["a","b"]] ──────────────────────

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, ast.Store):
            return
        if isinstance(node.value, ast.Constant):
            return
        if (isinstance(node.value, ast.Name)
                and node.value.id in self._CONFIG_NAMES):
            self.generic_visit(node)
            return
        if isinstance(node.slice, ast.Name):
            resolved = self._local_lists.get(node.slice.id)
            if resolved:
                self.cols_read.update(resolved)
                return
        self._add_col_read(node.slice)
        if isinstance(node.slice, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
            self._collect_filter(node.slice)
        self.generic_visit(node)

    # ── call analysis ────────────────────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> None:
        fname = _func_name(node)

        # F.col("x") / col("x") — a bare column reference
        if fname in _COL_FUNCS and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                self._add_col_read(first)

        # F.udf / F.pandas_udf
        if fname in _UDF_OPS:
            self.is_udf_candidate = True

        if not isinstance(node.func, ast.Attribute):
            # function-style call (col(...), when(...), udf(...)) — no method chain
            self._dispatch_function(fname, node)
            self.generic_visit(node)
            return

        attr = node.func.attr
        self.all_methods.add(attr)

        # ── plain string column-arg methods: select/drop/groupBy/orderBy… ──
        if attr in _COL_ARG_METHODS:
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    resolved = self._local_lists.get(arg.id)
                    if resolved:
                        self.cols_read.update(resolved)
                else:
                    self._add_col_read(arg)

        # ── column creation / renaming ──────────────────────────────────
        if attr == "withColumn" and node.args:
            self._add_col_written(node.args[0])
            if len(node.args) > 1:
                self.visit(node.args[1])
        if attr in ("withColumnRenamed",) and len(node.args) >= 2:
            self._add_col_read(node.args[0])
            self._add_col_written(node.args[1])
        if attr in ("withColumns", "withColumnsRenamed") and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Dict):
                if attr == "withColumns":
                    for k in arg.keys:
                        self._add_col_written(k)
                else:  # withColumnsRenamed({old: new})
                    for k in arg.keys:
                        self._add_col_read(k)
                    for v in arg.values:
                        self._add_col_written(v)

        # ── .alias("x") / .name("x") — output column (not table alias) ───
        if attr in ("alias", "name") and node.args:
            if _is_column_expr(node.func.value):
                self._add_col_written(node.args[0])

        # ── joins ────────────────────────────────────────────────────────
        if attr in ("join", "crossJoin"):
            self._collect_join(node, attr)

        # ── group by / rollup / cube ─────────────────────────────────────
        if attr in ("groupBy", "groupby", "rollup", "cube"):
            self.has_groupby = True

        # ── agg ──────────────────────────────────────────────────────────
        if attr == "agg":
            self._collect_agg(node)

        # ── functions.* aggregates used directly (F.sum("x")) ────────────
        if attr in _AGG_FUNCS and isinstance(node.func, ast.Attribute):
            for a in node.args:
                self._add_col_read(a)

        # ── window functions ─────────────────────────────────────────────
        if attr == "partitionBy":
            self.has_window_op = True
            for a in node.args:
                self._add_col_read(a)
        if attr == "over":
            self.has_window_op = True
            win_kind = "window"
            inner = node.func.value
            if isinstance(inner, ast.Call):
                ifn = _func_name(inner)
                if ifn in _RANKING_FUNCS:
                    win_kind = ifn
                elif ifn in _AGG_FUNCS:
                    win_kind = f"windowed_{ifn}"
                for a in inner.args:
                    self._add_col_read(a)
            self.window_funcs.append({
                "type": win_kind, "raw": ast.unparse(node)[:160],
            })
        if attr == "rangeBetween":
            self.has_rolling = True
            self.has_window_op = True
            self.window_funcs.append({
                "type": "range_window", "raw": ast.unparse(node)[:120],
            })
        if attr == "rowsBetween":
            self.has_window_op = True
            self.window_funcs.append({
                "type": "rows_window", "raw": ast.unparse(node)[:120],
            })
        if attr in _RANKING_FUNCS and isinstance(node.func, ast.Attribute):
            self.has_window_op = True
            entry: dict[str, Any] = {"type": attr, "raw": ast.unparse(node)[:120]}
            if attr in ("lag", "lead"):
                if node.args:
                    self._add_col_read(node.args[0])
                entry["offset"] = _safe_eval(node.args[1]) if len(node.args) > 1 else None
            self.window_funcs.append(entry)

        # ── filters ──────────────────────────────────────────────────────
        if attr in ("filter", "where"):
            arg = node.args[0] if node.args else None
            if isinstance(arg, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
                self._collect_filter(arg)
            expr = _safe_eval(arg) if isinstance(arg, ast.Constant) else None
            self.filters.append({"type": attr, "expression": expr,
                                  "raw": ast.unparse(node)[:120]})
        if attr == "isin":
            vals = [_safe_eval(a) for a in node.args] if node.args else None
            self.filters.append({"type": "isin", "values": vals,
                                  "raw": ast.unparse(node)[:120]})
        if attr == "between":
            self.filters.append({
                "type": "between",
                "low":  _safe_eval(node.args[0]) if len(node.args) > 0 else None,
                "high": _safe_eval(node.args[1]) if len(node.args) > 1 else None,
                "raw":  ast.unparse(node)[:120],
            })
        if attr in ("isNull", "isNotNull") and isinstance(node.func, ast.Attribute):
            col = _col_ref(node.func.value)
            self.filters.append({"type": attr, "column": col,
                                  "raw": ast.unparse(node)[:120]})
            if col:
                self.cols_read.add(col)

        # ── null / na handling ─────────────────────────────────────────────
        if attr in ("fillna", "fill", "dropna"):
            self.null_ops.add(attr)
            subset_node = _kw(node, "subset")
            if subset_node:
                self._add_col_read(subset_node)
                if attr == "dropna":
                    self.filters.append({
                        "type": "dropna", "subset": _str_vals(subset_node),
                        "raw": ast.unparse(node)[:120]})
        if attr == "drop":
            # df.na.drop(...)  vs  df.drop("colA")
            if (isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "na"):
                self.null_ops.add("dropna")
            # (column drops already captured via _COL_ARG_METHODS)

        # ── string / type ops ──────────────────────────────────────────────
        if attr in _STRING_OPS:  self.string_ops.add(attr)
        if attr in _NULL_OPS:    self.null_ops.add(attr)
        if attr in _TYPE_OPS:    self.type_ops.add(attr)

        # ── sorting ────────────────────────────────────────────────────────
        if attr in _SORT_OPS and not self._in_window_chain(node):
            self.sort_ops.add(attr)
            for a in node.args:
                self._add_col_read(a)

        # ── dedup ──────────────────────────────────────────────────────────
        if attr in _DEDUP_OPS:
            self.dedup_ops.add(attr)
            for a in node.args:
                self._add_col_read(a)

        # ── reshape ──────────────────────────────────────────────────────────
        if attr in _RESHAPE_OPS:
            self.reshape_ops.add(attr)
            if attr in ("explode", "explode_outer", "posexplode",
                        "posexplode_outer"):
                self.has_explode = True
                for a in node.args:
                    self._add_col_read(a)
            if attr == "pivot":
                self.has_pivot = True
                for a in node.args:
                    self._add_col_read(a)

        # ── union ──────────────────────────────────────────────────────────
        # (recognised in _classify via all_methods)

        # ── apply-style / pandas UDF application ────────────────────────────
        if attr in _APPLY_OPS:
            self.has_apply = True
            if attr in ("applyInPandas", "mapInPandas", "applyInArrow",
                        "mapInArrow"):
                self.is_udf_candidate = True

        self.generic_visit(node)

    def _dispatch_function(self, fname: "str | None", node: ast.Call) -> None:
        """Function-style (non-method) calls we still care about."""
        if fname is None:
            return
        if fname in _STRING_OPS:
            self.string_ops.add(fname)
        if fname in _NULL_OPS:
            self.null_ops.add(fname)
        if fname in _TYPE_OPS:
            self.type_ops.add(fname)
        if fname in _RESHAPE_OPS:
            self.reshape_ops.add(fname)
            if fname.startswith("explode") or fname.startswith("posexplode"):
                self.has_explode = True
                for a in node.args:
                    self._add_col_read(a)

    def _in_window_chain(self, node: ast.Call) -> bool:
        """True if this call is part of a Window.partitionBy(...).orderBy(...)
        chain rather than a DataFrame sort."""
        cur = node.func.value if isinstance(node.func, ast.Attribute) else None
        while isinstance(cur, ast.Call):
            f = cur.func
            if isinstance(f, ast.Name) and f.id == "Window":
                return True
            if isinstance(f, ast.Attribute):
                if f.attr in ("partitionBy", "rangeBetween", "rowsBetween"):
                    return True
                if isinstance(f.value, ast.Name) and f.value.id == "Window":
                    return True
                cur = f.value
            else:
                break
        if isinstance(cur, ast.Name) and cur.id == "Window":
            return True
        return False

    # ── filter collection ────────────────────────────────────────────────

    def _collect_filter(self, node: "ast.expr") -> None:
        if isinstance(node, ast.Compare):
            col = _col_ref(node.left)
            for op_node, comparator in zip(node.ops, node.comparators):
                op_str = _OP_NAMES.get(type(op_node), type(op_node).__name__)
                value  = _safe_eval(comparator)
                self.filters.append({
                    "type": "comparison", "column": col,
                    "op": op_str, "value": value,
                    "raw": ast.unparse(node)[:120],
                })
                if col:
                    self.cols_read.add(col)
        elif isinstance(node, ast.BoolOp):
            for val in node.values:
                self._collect_filter(val)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert)):
            self._collect_filter(node.operand)

    # ── join collection ──────────────────────────────────────────────────

    def _collect_join(self, node: ast.Call, join_type: str) -> None:
        # PySpark: join(other, on=None, how=None) — on is arg[1], how is arg[2].
        how = _safe_eval(_kw(node, "how"))
        if how is None and len(node.args) > 2:
            how = _safe_eval(node.args[2])
        info: dict[str, Any] = {
            "join_method": join_type,
            "how":         how or ("cross" if join_type == "crossJoin" else "inner"),
            "on":          None, "left_on": None,
            "right_on":    None, "by":      None,
            "direction":   None,
            "indicator":   None,
            "suffixes":    None,
            "right_table": ast.unparse(node.args[0])[:80] if node.args else None,
            "raw":         ast.unparse(node)[:200],
        }
        # PySpark join key is the 2nd positional arg or the `on=` kwarg.
        on_node = _kw(node, "on")
        if on_node is None and len(node.args) > 1:
            on_node = node.args[1]
        if on_node is not None:
            strs = _str_vals(on_node)
            if strs:
                info["on"] = strs[0] if len(strs) == 1 else strs
                self._add_col_read(on_node)
            else:
                # expression join: df.a == df2.b  → capture referenced columns
                for child in ast.walk(on_node):
                    c = None
                    if isinstance(child, (ast.Call, ast.Subscript, ast.Attribute)):
                        c = _col_ref(child)
                    if c:
                        self.cols_read.add(c)
        self.joins.append(info)

    # ── agg collection ──────────────────────────────────────────────────

    def _collect_agg(self, node: ast.Call) -> None:
        funcs_used:  set[str] = set()
        input_cols:  set[str] = set()
        output_cols: set[str] = set()

        # dict form: .agg({"amount": "sum"})
        if node.args and isinstance(node.args[0], ast.Dict):
            arg = node.args[0]
            for k, v in zip(arg.keys, arg.values):
                src = _str_vals(k)
                fn  = _str_vals(v)
                input_cols.update(c for c in (_clean_col(s) for s in src) if c)
                funcs_used.update(fn)
                self._add_col_read(k)
        else:
            # expression form: F.sum("amount").alias("total"), ...
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                fn, in_cols, out = self._parse_agg_expr(arg)
                if fn:
                    funcs_used.add(fn)
                input_cols.update(in_cols)
                if out:
                    output_cols.add(out)

        groupby_keys = self._groupby_keys(node.func.value)

        for c in output_cols:
            self.cols_written.add(c)
        for c in input_cols:
            self.cols_read.add(c)
        self.aggregations.append({
            "groupby_keys":   groupby_keys,
            "agg_functions":  sorted(funcs_used),
            "input_columns":  sorted(input_cols),
            "output_columns": sorted(output_cols),
            "raw":            ast.unparse(node)[:200],
        })

    def _parse_agg_expr(self, expr: Any) -> tuple["str | None", list[str], "str | None"]:
        """Parse F.sum("amount").alias("total") -> ('sum', ['amount'], 'total')."""
        out = None
        cur = expr
        # peel .alias / .name wrappers (outermost)
        while (isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute)
               and cur.func.attr in ("alias", "name")):
            if cur.args and isinstance(cur.args[0], ast.Constant):
                out = cur.args[0].value
            cur = cur.func.value
        # peel .cast / trailing transforms to find the aggregate
        fn = None
        in_cols: list[str] = []
        for child in ast.walk(cur) if isinstance(cur, ast.AST) else []:
            if isinstance(child, ast.Call) and _func_name(child) in _AGG_FUNCS:
                fn = _func_name(child)
                for a in child.args:
                    in_cols.extend(c for c in (_clean_col(s) for s in _str_vals(a)) if c)
                    c = _col_ref(a)
                    if c:
                        in_cols.append(c)
                break
        return fn, list(dict.fromkeys(in_cols)), out

    def _groupby_keys(self, node: Any) -> list[str]:
        """Walk back a chain to find the groupBy/rollup/cube keys."""
        cur = node
        while isinstance(cur, ast.Call):
            if (isinstance(cur.func, ast.Attribute)
                    and cur.func.attr in ("groupBy", "groupby", "rollup", "cube")):
                keys: list[str] = []
                for a in cur.args:
                    keys.extend(c for c in (_clean_col(s) for s in _str_vals(a)) if c)
                    c = _col_ref(a)
                    if c:
                        keys.append(c)
                return list(dict.fromkeys(keys))
            cur = cur.func.value if isinstance(cur.func, ast.Attribute) else None
        return []


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify(func: dict) -> tuple[list[str], str, str, list[str]]:
    ops     = set(func.get("all_methods", []))
    joins   = func.get("joins", [])
    aggs    = func.get("aggregations", [])
    windows = func.get("window_functions", [])
    filters = func.get("filters", [])

    has_rolling = func.get("has_rolling", False)
    has_window  = func.get("has_window_op", False)
    has_explode = func.get("has_explode", False)
    has_pivot   = func.get("has_pivot", False)
    is_udf      = func.get("is_udf_candidate", False)
    has_groupby = func.get("has_groupby", False)
    has_apply   = func.get("has_apply", False)
    has_dedup   = bool(func.get("dedup_operations"))
    has_union   = bool(ops & _UNION_OPS)

    types: set[str] = set()
    if func.get("null_operations") or func.get("type_operations") or \
       func.get("string_operations"):
        types.add("cleaning")
    if func.get("type_operations"):
        types.add("type_casting")
    if func.get("string_operations"):
        types.add("string_normalization")
    if joins:
        types.add("join_enrichment")
    if has_groupby and (aggs or (ops & _AGG_FUNCS)):
        types.add("aggregation")
    if has_rolling or has_window or windows:
        types.add("window_computation")
    if ops & {"when", "otherwise", "expr", "selectExpr"}:
        types.add("conditional_derivation")
    if filters or (ops & {"filter", "where", "isin", "dropna"}):
        types.add("filtering")
    if has_explode or has_pivot or func.get("reshape_operations"):
        types.add("reshaping")
    if has_dedup:
        types.add("deduplication")
    if has_union:
        types.add("union")
    if is_udf:
        types.add("custom_transformation")
    if has_apply and not is_udf:
        types.add("window_computation")
    if func.get("sort_operations"):
        types.add("sorting")
    if not types:
        types.add("other")

    patterns: set[str] = set()
    risk = "low"
    warnings: list[str] = []

    if is_udf:
        patterns.add("udf_candidate")
        if risk == "low": risk = "medium"
        warnings.append(
            "Python UDF (F.udf) detected — serialization-bound and opaque to "
            "Catalyst. Prefer native F.* expressions or F.pandas_udf where possible."
        )
    if has_rolling:
        patterns.add("window_function")
        if risk == "low": risk = "medium"
        warnings.append(
            "rangeBetween window detected — ensure the ordering column is cast "
            "to a numeric/long (e.g. ts.cast('long')) so the range is in the "
            "intended units."
        )
    if (has_window or windows) and "window_function" not in patterns:
        patterns.add("window_function")
        if risk == "low": risk = "medium"
        if not any(w.get("type") == "window" or "partitionBy" in func.get("all_methods", [])
                   for w in windows):
            warnings.append(
                "Window without partitionBy shuffles all rows to a single "
                "partition — verify a partition key is present."
            )
    if joins:
        patterns.add("join")
        warnings.append("Broadcast small dimension tables: F.broadcast(dim_df).")
    if has_groupby and aggs:
        patterns.add("aggregation")
    if aggs and not has_groupby:
        patterns.add("two_phase")
    if has_explode:
        patterns.add("explode")
        warnings.append(
            "F.explode drops rows with null/empty arrays. Use F.explode_outer "
            "to retain them if the original pandas .explode() semantics require it."
        )
    if has_union:
        patterns.add("union")
        warnings.append(
            "Use unionByName(allowMissingColumns=…) rather than union() when "
            "column order/set may differ between frames."
        )
    if has_pivot:
        patterns.add("aggregation")
    if not patterns:
        patterns.add("pure_map")

    final_pattern = next(iter(patterns)) if len(patterns) == 1 else "mixed"
    if final_pattern == "mixed" and risk == "low":
        risk = "medium"
    return sorted(types), final_pattern, risk, warnings


# ---------------------------------------------------------------------------
# Module-level collectors (imports / constants / classes / call graph)
# ---------------------------------------------------------------------------

def _collect_imports(tree: ast.Module) -> dict:
    _KNOWN_STDLIB = {
        "hashlib", "logging", "sys", "os", "re", "json", "csv", "io",
        "datetime", "collections", "itertools", "functools", "pathlib",
        "typing", "abc", "math", "random", "copy", "time", "warnings",
        "traceback", "inspect", "importlib", "contextlib", "dataclasses",
        "enum", "struct", "gc", "operator", "string", "textwrap",
        "__future__", "argparse", "subprocess", "threading", "multiprocessing",
    }
    _KNOWN_THIRD = {
        "pandas", "numpy", "scipy", "sklearn", "matplotlib", "seaborn",
        "pyspark", "pyarrow", "fastparquet", "boto3", "requests",
        "sqlalchemy", "psycopg2", "pymysql", "google", "azure",
        "databricks", "delta", "mlflow", "optuna", "torch", "tensorflow",
        "keras", "xgboost", "lightgbm", "catboost", "statsmodels",
        "snowflake", "snowpark",
    }
    stdlib, third, local = [], [], []
    seen: set[str] = set()

    def _cat(node: Any) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                key = f"import:{alias.name}:{alias.asname}"
                if key in seen: return
                seen.add(key)
                base  = alias.name.split(".")[0]
                entry = {"module": alias.name, "alias": alias.asname}
                (stdlib if base in _KNOWN_STDLIB
                 else third if base in _KNOWN_THIRD
                 else local).append(entry)
        else:
            base  = (node.module or "").split(".")[0]
            names = [{"name": a.name, "alias": a.asname} for a in node.names]
            key   = f"from:{node.module}:{node.level}:{','.join(n['name'] for n in names)}"
            if key in seen: return
            seen.add(key)
            entry = {"module": node.module, "names": names, "level": node.level}
            (stdlib if base in _KNOWN_STDLIB
             else third if base in _KNOWN_THIRD
             else local).append(entry)

    def _is_main_guard(node: ast.AST) -> bool:
        if not isinstance(node, ast.If): return False
        t = node.test
        return (isinstance(t, ast.Compare)
                and isinstance(t.left, ast.Name) and t.left.id == "__name__"
                and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)
                and len(t.comparators) == 1
                and isinstance(t.comparators[0], ast.Constant)
                and t.comparators[0].value == "__main__")

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _cat(node)
        elif _is_main_guard(node):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    _cat(child)
        elif isinstance(node, ast.Try):
            for child in node.body + node.handlers:
                sub_nodes = child.body if isinstance(child, ast.ExceptHandler) else [child]
                for sub in sub_nodes:
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        _cat(sub)
    return {"standard_library": stdlib, "third_party": third, "local": local}


def _collect_constants(tree: ast.Module) -> dict:
    consts: dict = {}
    for node in tree.body:
        value_node = name = None
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name, value_node = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value_node = node.target.id, node.value
        if name and value_node and name.isupper():
            val = _safe_eval(value_node)
            if isinstance(val, set):
                val = sorted(val, key=str)
            consts[name] = val
    return consts


def _extract_tag_and_comment(
    line_no: int,
    src_lines: list[str],
    prev_def_line: int,
) -> tuple[str | None, str]:
    """Scan only from prev_def_line (exclusive) to line_no (exclusive) so tags
    from the previous function don't leak into the next one."""
    tag, note = None, ""
    start = max(0, prev_def_line)
    end   = max(0, line_no - 1)
    block = src_lines[start:end]
    for line in block:
        s = line.strip()
        m = re.search(r"\[(T\d{2,3})\]", s)
        if m:
            tag = m.group(1)
        if ("PySpark" in s or "Snowpark" in s) and (
                "equivalent" in s or "PySpark:" in s or "Snowpark:" in s):
            note = re.sub(r"^#\s*", "", s).strip()
    return tag, note


def _collect_classes(tree: ast.Module) -> list[dict]:
    classes = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = [
            {
                "name":       item.name,
                "line":       item.lineno,
                "docstring":  ast.get_docstring(item),
                "parameters": [a.arg for a in item.args.args if a.arg != "self"],
            }
            for item in node.body if isinstance(item, ast.FunctionDef)
        ]
        classes.append({
            "name":      node.name,
            "bases":     [ast.unparse(b) for b in node.bases],
            "line":      node.lineno,
            "docstring": ast.get_docstring(node),
            "methods":   methods,
        })
    return classes


def _call_graph(tree: ast.Module, all_names: set[str]) -> dict:
    graph: dict[str, list[str]] = {}

    def _scan_body(key: str, stmts: list[ast.stmt]) -> None:
        calls: list[str] = []
        for stmt in stmts:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Call):
                    n = None
                    if isinstance(child.func, ast.Name):
                        n = child.func.id
                    elif isinstance(child.func, ast.Attribute):
                        n = child.func.attr
                    if n and n in all_names and n != key and n not in calls:
                        calls.append(n)
        if calls:
            graph[key] = calls

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            _scan_body(node.name, node.body)
        elif isinstance(node, ast.If):
            t = node.test
            is_main = (isinstance(t, ast.Compare)
                       and isinstance(t.left, ast.Name) and t.left.id == "__name__"
                       and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)
                       and len(t.comparators) == 1
                       and isinstance(t.comparators[0], ast.Constant)
                       and t.comparators[0].value == "__main__")
            if is_main:
                _scan_body("__main__", node.body)
    return graph


def _resolve_local_import(
    module_name: str,
    base_dir: Path,
    level: int = 0,
) -> tuple[Path | None, list[str]]:
    """Resolve a local import to a .py file. Returns (resolved_path, tried_paths)."""
    base_dir = Path(base_dir)
    if not module_name and level == 0:
        return None, []

    search_dir = base_dir
    for _ in range(max(level - 1, 0)):
        search_dir = search_dir.parent

    parts    = module_name.replace(".", "/")
    basename = module_name.split(".")[-1]
    head     = module_name.split(".")[0]

    roots = [search_dir, search_dir.parent, search_dir.parent.parent]
    if search_dir.name == head:
        roots.append(search_dir.parent)

    tried: list[str] = []
    for root in roots:
        for cand in (
            root / f"{parts}.py",
            root / parts / "__init__.py",
            root / f"{basename}.py",
        ):
            c = str(cand)
            if c not in tried:
                tried.append(c)
            if cand.exists():
                return cand.resolve(), tried
    return None, tried


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pipeline(
    script_path: str,
    follow_imports: bool = False,
    _visited: set | None = None,
) -> dict:
    """Parse a PySpark data-transformation pipeline and return a comprehensive
    JSON-serialisable analysis dict — identical schema to the pandas parser."""
    if _visited is None:
        _visited = set()
    abs_path = str(Path(script_path).resolve())
    if abs_path in _visited:
        return {}
    _visited.add(abs_path)

    path   = Path(script_path)
    source = path.read_text(encoding="utf-8")
    lines  = source.splitlines()
    tree   = ast.parse(source, filename=str(path))

    metadata = {
        "script_path":      str(path.resolve()),
        "parsed_at":        datetime.now(timezone.utc).isoformat(),
        "total_lines":      len(lines),
        "module_docstring": ast.get_docstring(tree),
    }

    imports   = _collect_imports(tree)
    constants = _collect_constants(tree)
    classes   = _collect_classes(tree)

    all_func_names: set[str] = {
        n.name for n in tree.body if isinstance(n, ast.FunctionDef)
    }

    end_line_map: dict[int, int] = {}
    top_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for i, fn in enumerate(top_defs):
        prev_end = top_defs[i - 1].end_lineno if i > 0 else 0
        end_line_map[fn.lineno] = prev_end

    functions: list[dict] = []
    for func_node in tree.body:
        if not isinstance(func_node, ast.FunctionDef):
            continue
        if func_node.name.startswith("__"):
            continue

        prev_end_1indexed = end_line_map.get(func_node.lineno, 0)
        tag, pyspark_comment = _extract_tag_and_comment(
            func_node.lineno, lines, prev_end_1indexed
        )

        params = [
            {"name": a.arg,
             "annotation": ast.unparse(a.annotation) if a.annotation else None}
            for a in func_node.args.args
        ]
        return_ann = ast.unparse(func_node.returns) if func_node.returns else None

        an = _FunctionAnalyser(lines)
        an.visit(func_node)

        partial: dict = {
            "name":              func_node.name,
            "tag":               tag,
            "line_start":        func_node.lineno,
            "line_end":          getattr(func_node, "end_lineno", None),
            "docstring":         ast.get_docstring(func_node),
            "parameters":        params,
            "return_annotation": return_ann,
            "columns": {
                "read":    sorted(an.cols_read),
                "written": sorted(an.cols_written),
                "new":     sorted(an.cols_written - an.cols_read),
                "all":     sorted(an.cols_read | an.cols_written),
            },
            "joins":            an.joins,
            "aggregations":     an.aggregations,
            "filters":          an.filters,
            "window_functions": an.window_funcs,
            "operations": {
                "string":        sorted(an.string_ops),
                "null_handling": sorted(an.null_ops),
                "type_casting":  sorted(an.type_ops),
                "sorting":       sorted(an.sort_ops),
                "deduplication": sorted(an.dedup_ops),
                "reshaping":     sorted(an.reshape_ops),
                "all_methods":   sorted(an.all_methods),
            },
            "flags": {
                "has_apply":        an.has_apply,
                "is_udf_candidate": an.is_udf_candidate,
                "has_merge_asof":   an.has_merge_asof,
                "has_rolling":      an.has_rolling,
                "has_window_op":    an.has_window_op,
                "has_explode":      an.has_explode,
                "has_pivot":        an.has_pivot,
                "has_groupby":      an.has_groupby,
            },
            "transformation_types": [],
            "pyspark": {},
        }

        t_types, pattern, risk, warnings = _classify({
            **partial["flags"],
            **{k: sorted(v) for k, v in partial["operations"].items()},
            "joins":              partial["joins"],
            "aggregations":       partial["aggregations"],
            "window_functions":   partial["window_functions"],
            "filters":            partial["filters"],
            "all_methods":        partial["operations"]["all_methods"],
            "null_operations":    partial["operations"]["null_handling"],
            "type_operations":    partial["operations"]["type_casting"],
            "string_operations":  partial["operations"]["string"],
            "sort_operations":    partial["operations"]["sorting"],
            "dedup_operations":   partial["operations"]["deduplication"],
            "reshape_operations": partial["operations"]["reshaping"],
        })
        partial["transformation_types"] = t_types
        partial["pyspark"] = {
            "pattern":  pattern,
            "risk":     risk,
            "comment":  pyspark_comment,
            "warnings": warnings,
        }
        functions.append(partial)

    # ── summary ──────────────────────────────────────────────────────────
    all_columns: set[str] = set()
    type_dist:   dict[str, int] = {}
    risk_dist    = {"low": 0, "medium": 0, "high": 0}
    all_jt:      set[str] = set()
    all_af:      set[str] = set()
    udfs: list[str] = []
    high: list[str] = []

    for f in functions:
        all_columns.update(f["columns"]["all"])
        for t in f["transformation_types"]:
            type_dist[t] = type_dist.get(t, 0) + 1
        r = f["pyspark"]["risk"]
        risk_dist[r] += 1
        for j in f["joins"]:
            all_jt.add(f"{j['join_method']}:{j['how']}")
        for a in f["aggregations"]:
            all_af.update(a["agg_functions"])
        if f["flags"]["is_udf_candidate"]:
            udfs.append(f["name"])
        if f["pyspark"]["risk"] == "high":
            high.append(f["name"])

    result = {
        "metadata":   metadata,
        "imports":    imports,
        "constants":  constants,
        "classes":    classes,
        "functions":  functions,
        "call_graph": _call_graph(tree, all_func_names),
        "summary": {
            "function_count":                   len(functions),
            "total_columns_referenced":         len(all_columns),
            "all_columns_referenced":           sorted(all_columns),
            "transformation_type_distribution": type_dist,
            "pyspark_risk_distribution":        risk_dist,
            "all_join_types":                   sorted(all_jt),
            "all_aggregation_functions":        sorted(all_af),
            "udf_candidates":                   udfs,
            "high_risk_functions":              high,
        },
        "dependencies":        {},
        "unresolved_imports":  [],
    }

    # ── follow local imports ─────────────────────────────────────────────
    if follow_imports:
        base_dir = path.parent
        for imp in imports.get("local", []):
            module_name = imp.get("module") or ""
            level       = imp.get("level", 0)

            dep_path, tried = _resolve_local_import(module_name, base_dir, level)

            if dep_path is None:
                msg = (f"[follow-imports] Could not resolve local import "
                       f"'{module_name}' — tried:\n"
                       + "\n".join(f"    {p}" for p in tried))
                print(msg, file=sys.stderr)
                result["unresolved_imports"].append({
                    "module": module_name,
                    "tried":  tried,
                })
                continue

            dep_key = str(dep_path)
            if dep_key in _visited:
                result["dependencies"][dep_path.name] = {
                    "note": f"Already parsed (circular): {dep_path}"
                }
                continue

            try:
                dep_result = parse_pipeline(
                    str(dep_path),
                    follow_imports=follow_imports,
                    _visited=_visited,
                )
                result["dependencies"][dep_path.name] = dep_result
                print(f"  Followed import -> {dep_path.name}  "
                      f"({dep_result.get('summary', {}).get('function_count', '?')} functions)",
                      file=sys.stderr)
            except SyntaxError as exc:
                result["dependencies"][dep_path.name] = {
                    "error": f"SyntaxError in {dep_path}: {exc}"}
            except Exception as exc:
                result["dependencies"][dep_path.name] = {
                    "error": f"Failed to parse {dep_path}: {type(exc).__name__}: {exc}"}

        dep_func_count = sum(
            d.get("summary", {}).get("function_count", 0)
            for d in result["dependencies"].values()
            if isinstance(d, dict) and "summary" in d
        )
        dep_high_risk = [
            f"{fname}  ({dep_name})"
            for dep_name, d in result["dependencies"].items()
            if isinstance(d, dict) and "summary" in d
            for fname in d["summary"].get("high_risk_functions", [])
        ]
        dep_udf_cands = [
            f"{fname}  ({dep_name})"
            for dep_name, d in result["dependencies"].items()
            if isinstance(d, dict) and "summary" in d
            for fname in d["summary"].get("udf_candidates", [])
        ]
        result["summary"]["dependencies_parsed"]         = len(result["dependencies"])
        result["summary"]["dependencies_function_count"] = dep_func_count
        result["summary"]["dependencies_high_risk"]      = dep_high_risk
        result["summary"]["dependencies_udf_candidates"] = dep_udf_cands

    return result


# ---------------------------------------------------------------------------
# CLI / summary printing
# ---------------------------------------------------------------------------

def _wrap(items: list[str], indent: str = "      ", width: int = 100) -> str:
    if not items:
        return f"{indent}—"
    lines: list[str] = []
    current: list[str] = []

    def _rendered() -> str:
        return indent + ", ".join(current)

    for item in items:
        if not current:
            current.append(item)
        elif len(_rendered()) + 2 + len(item) <= width:
            current.append(item)
        else:
            lines.append(_rendered())
            current = [item]
    if current:
        lines.append(_rendered())
    return "\n".join(lines)


def _print_module_summary(r: dict, out: Any, label: str = "") -> None:
    s  = r.get("summary", {})
    md = r.get("metadata", {})
    sep = "=" * 80

    print(f"\n{sep}", file=out)
    if label:
        print(f"  [{label}]", file=out)
    if md.get("script_path"):
        print(f"  {md['script_path']}", file=out)
    if md.get("parsed_at"):
        print(f"  Parsed : {md['parsed_at']}   Lines: {md.get('total_lines', '?')}",
              file=out)
    print(sep, file=out)

    print(f"\n  Functions  : {s.get('function_count', 0)}", file=out)
    print(f"  Columns    : {s.get('total_columns_referenced', 0)} unique", file=out)
    print(f"  Risk dist  : {s.get('pyspark_risk_distribution', {})}", file=out)
    print(f"  High-risk  : {s.get('high_risk_functions', [])}", file=out)
    print(f"  UDF cands  : {s.get('udf_candidates', [])}", file=out)
    print(f"  Join types : {s.get('all_join_types', [])}", file=out)
    print(f"  Agg funcs  : {s.get('all_aggregation_functions', [])}", file=out)

    unresolved = r.get("unresolved_imports", [])
    if unresolved:
        print(f"\n  Unresolved imports ({len(unresolved)}):", file=out)
        for u in unresolved:
            print(f"    ✗  {u['module']}", file=out)
            for p in u.get("tried", []):
                print(f"         tried: {p}", file=out)

    functions = r.get("functions", [])
    if functions:
        print(f"\n  {'─' * 78}", file=out)
        print(f"  FUNCTIONS ({len(functions)})", file=out)
        print(f"  {'─' * 78}", file=out)

    for f in functions:
        tag     = f"  [{f['tag']}]" if f.get("tag") else "  "
        risk    = f["pyspark"]["risk"].upper()
        pattern = f["pyspark"]["pattern"]
        types   = ", ".join(f.get("transformation_types", [])) or "other"
        reads   = f["columns"].get("read", [])
        new_c   = f["columns"].get("new", [])
        written = f["columns"].get("written", [])

        if new_c:
            out_cols = new_c
            out_label = "cols written (new)"
        elif written:
            out_cols  = [f"~{c}" for c in written]
            out_label = "cols modified"
        else:
            out_cols  = []
            out_label = "cols written"

        print(f"\n{tag}  {f['name']}  "
              f"[{pattern}]  [{risk}]  L{f['line_start']}–{f['line_end']}",
              file=out)
        print(f"      types      : {types}", file=out)

        if reads:
            print(f"      cols read  :", file=out)
            print(_wrap(reads), file=out)
        else:
            print(f"      cols read  : —", file=out)

        if out_cols:
            print(f"      {out_label:<18}:", file=out)
            print(_wrap(out_cols), file=out)
        else:
            print(f"      {out_label:<18}: —", file=out)

        for w in f["pyspark"].get("warnings", []):
            print(f"      ⚠  {w}", file=out)

    all_cols = s.get("all_columns_referenced", [])
    if all_cols:
        print(f"\n  All columns referenced ({len(all_cols)}):", file=out)
        print(_wrap(all_cols, indent="    ", width=100), file=out)


def _print_summary(r: dict, out: Any = None) -> None:
    if out is None:
        out = sys.stderr

    _print_module_summary(r, out=out, label="MAIN FILE")

    deps = r.get("dependencies", {})
    if deps:
        print(f"\n\n{'=' * 80}", file=out)
        print(f"  DEPENDENCIES ({len(deps)})", file=out)
        print(f"{'=' * 80}", file=out)
        for dep_name, dep_data in deps.items():
            if "error" in dep_data:
                print(f"\n  ✗  {dep_name}", file=out)
                print(f"     {dep_data['error']}", file=out)
            elif "note" in dep_data:
                print(f"\n  ↩  {dep_name}  (skipped — {dep_data['note']})", file=out)
            else:
                _print_module_summary(dep_data, out=out,
                                      label=f"DEPENDENCY: {dep_name}")
    print(file=out)


def run_parser(script_path: str, follow_imports: bool = False,
               print_summary: bool = False) -> dict:
    """Parse a PySpark pipeline and RETURN the analysis dict.

    Nothing is written to disk — the caller (e.g. a pre-agent-call callback)
    consumes the returned JSON directly, e.g. to pull out function names and
    stash them in the agent state.

    Set print_summary=True to also emit the human-readable summary to stderr.
    """
    print(f"Parsing PySpark pipeline: {script_path} with follow_imports={follow_imports}",
          file=sys.stderr)
    result = parse_pipeline(script_path, follow_imports=follow_imports)

    if print_summary:
        buf = io.StringIO()
        _print_summary(result, out=buf)
        sys.stderr.write(buf.getvalue())
        sys.stderr.flush()

    return result


def extract_function_names(result: dict, include_dependencies: bool = True) -> list[str]:
    """Pull every parsed function name out of a run_parser() result.

    Convenience for the pre-agent-call callback that needs to set the list of
    PySpark function names in the agent state before the agent runs.
    """
    names: list[str] = [f["name"] for f in result.get("functions", [])]
    if include_dependencies:
        for dep in result.get("dependencies", {}).values():
            if isinstance(dep, dict) and "functions" in dep:
                names.extend(f["name"] for f in dep["functions"])
    # de-dupe, preserve order
    return list(dict.fromkeys(names))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Comprehensive AST parser for PySpark data-transformation pipelines."
    )
    ap.add_argument("script", help="Path to the PySpark pipeline file to parse.")
    ap.add_argument(
        "--follow-imports", action="store_true", default=False,
        help="Also parse every resolvable local dependency.",
    )
    ap.add_argument(
        "--summary", action="store_true", default=False,
        help="Print the human-readable summary to stderr as well.",
    )
    args = ap.parse_args()
    result = run_parser(args.script, follow_imports=args.follow_imports,
                        print_summary=args.summary)
    # emit the JSON on stdout so the CLI is still usable / pipeable
    print(json.dumps(result, indent=2))
