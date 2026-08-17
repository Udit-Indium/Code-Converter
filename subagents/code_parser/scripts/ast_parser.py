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
        if node.value.id == "np":
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
    JSON-serializable.  Sets arise from Python set literals like
    {"txn_id", "customer_id", ...} evaluated by ast.literal_eval."""
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

_STRING_OPS = {
    "lower", "upper", "title", "strip", "lstrip", "rstrip", "replace",
    "split", "rsplit", "contains", "startswith", "endswith", "extract",
    "findall", "match", "fullmatch", "get", "pad", "center", "ljust",
    "rjust", "zfill", "cat", "normalize", "encode", "decode",
    "capitalize", "swapcase",
}
_NULL_OPS = {
    "fillna", "dropna", "isna", "notna", "isnull", "notnull",
    "bfill", "ffill", "backfill", "pad",
}
_TYPE_OPS = {
    "astype", "to_numeric", "to_datetime", "to_timedelta",
    "infer_objects", "convert_dtypes", "cast",
}
_WINDOW_OPS = {
    "rolling", "expanding", "ewm", "shift", "diff",
    "rank", "pct_change", "cumsum", "cumprod", "cummax", "cummin",
}
_AGG_FUNCS = {
    "sum", "mean", "median", "min", "max", "std", "var",
    "count", "nunique", "size", "first", "last", "any", "all",
    "prod", "sem", "skew", "kurt", "quantile", "idxmin", "idxmax",
}
_SORT_OPS  = {"sort_values", "sort_index", "nlargest", "nsmallest", "argsort"}
_DEDUP_OPS = {"drop_duplicates", "duplicated"}
_RESHAPE_OPS = {
    "pivot_table", "pivot", "melt", "stack", "unstack",
    "explode", "crosstab", "transpose",
}

# Methods whose specific kwargs carry column names (and ONLY those kwargs
# should be fed to _add_read with context="col").
_COL_KWARG_MAP: dict[str, list[str]] = {
    "groupby":     ["by"],
    "sort_values": ["by"],
    "dropna":      ["subset"],
    "merge":       ["on", "left_on", "right_on"],
    "merge_asof":  ["on", "left_on", "right_on", "by"],
    "join":        ["on"],
    "pivot_table": ["index", "columns", "values"],
    "pivot":       ["index", "columns", "values"],
    "melt":        ["id_vars", "value_vars"],
    "reindex":     ["columns"],
    "drop":        ["columns", "labels"],
}


# ---------------------------------------------------------------------------
# Per-function analyser
# ---------------------------------------------------------------------------

class _FunctionAnalyser(ast.NodeVisitor):
    """
    Walk one FunctionDef body and collect all transformation metadata.

    FIX I-05: _add_col_read() is the only path that populates cols_read.
              It is called exclusively from subscript slices and known
              column-bearing kwargs — NOT from arbitrary string constants
              such as CONFIG["fraud_amount_threshold_usd"].

    FIX I-06: visit_FunctionDef intercepts nested function definitions,
              analyses their bodies for apply(axis=1), and merges the
              is_udf_candidate flag back to the parent.
    """

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

        self.has_apply        = False
        self.has_merge_asof   = False
        self.has_rolling      = False
        self.has_window_op    = False
        self.has_explode      = False
        self.has_pivot        = False
        self.is_udf_candidate = False
        self.has_groupby      = False

        # FIX MR-4: track local list assignments for variable-subscript resolution
        self._local_lists: dict[str, list[str]] = {}

        # track nesting depth so visit_FunctionDef handles inner helpers
        self._depth = 0

    # ── column helpers (FIX I-05) ────────────────────────────────────────

    def _add_col_read(self, node: Any) -> None:
        """Add strings from `node` to cols_read.
        Only called from contexts that are definitively column references:
        subscript slices (df["col"]) and column-bearing kwargs (on=, by=…).
        """
        for s in _str_vals(node):
            self.cols_read.add(s)

    def _add_col_written(self, node: Any) -> None:
        for s in _str_vals(node):
            self.cols_written.add(s)

    # ── nested function handling ─────────────────────────────────────────

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Intercept inner helpers:
        - bubble up UDF/apply signals to parent  (FIX I-06)
        - bubble up subscript col reads from inner body (FIX MR-1)
        - capture row.get("col") as col reads     (FIX MR-2)
        """
        if self._depth == 0:
            self._depth += 1
            self.generic_visit(node)
            self._depth -= 1
        else:
            inner = _FunctionAnalyser(self.src_lines)
            inner._depth = 1
            inner._local_lists = dict(self._local_lists)  # inherit parent scope
            inner.generic_visit(node)
            # Bubble UDF flag
            if inner.is_udf_candidate or inner.has_apply:
                self.has_apply = True
                self.is_udf_candidate = True
            # FIX MR-1: bubble inner subscript reads to parent
            self.cols_read.update(inner.cols_read)
            # FIX MR-2: capture row.get("col_name") as col reads.
            # Inside a row-wise apply helper the frame is accessed via
            # row.get("col") / row["col"] rather than df["col"].
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    fn = child.func
                    if (isinstance(fn, ast.Attribute) and fn.attr == "get"
                            and child.args):
                        first = child.args[0]
                        if isinstance(first, ast.Constant) and isinstance(first.value, str):
                            self.cols_read.add(first.value)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    # ── assignment targets ───────────────────────────────────────────────

    def visit_Assign(self, node: ast.Assign) -> None:
        # FIX MR-4: track local list/variable assignments so we can resolve
        # them when used as subscript slices: cust_cols = ["a","b"]; df[cust_cols]
        for target in node.targets:
            if isinstance(target, ast.Name):
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    vals = [
                        elt.value for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
                    if vals:
                        self._local_lists[target.id] = vals
            # FP fix: only record write, don't let generic_visit re-visit the
            # subscript target as a read (Store context subscripts are writes only)
            if isinstance(target, ast.Subscript):
                self._add_col_written(target.slice)
        # Visit the RHS (value) but NOT the targets again — prevents written
        # col names from leaking into cols_read via visit_Subscript on Store nodes
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Subscript):
            self._add_col_written(node.target.slice)
        if node.value is not None:
            self.visit(node.value)

    # Names that are dictionaries/configs — subscript strings on these
    # are config keys, NOT DataFrame column names.
    _CONFIG_NAMES = {
        "CONFIG", "config", "TIER_THRESHOLDS", "tier_thresholds",
        "thr", "thresholds", "settings", "params", "kwargs",
    }

    # ── subscript reads: df["col"] / df[["a","b"]] ──────────────────────

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # Skip Store-context subscripts — handled by visit_Assign as writes.
        if isinstance(node.ctx, ast.Store):
            return

        if isinstance(node.value, ast.Constant):
            # string["something"] — not a DataFrame op
            return

        # Skip config/dict lookups
        if (isinstance(node.value, ast.Name)
                and node.value.id in self._CONFIG_NAMES):
            self.generic_visit(node)
            return

        # FIX MR-4: resolve variable-bound list used as subscript
        # e.g. cust_cols = ["a","b"]; df[cust_cols]
        if isinstance(node.slice, ast.Name):
            resolved = self._local_lists.get(node.slice.id)
            if resolved:
                self.cols_read.update(resolved)
                # Don't generic_visit — nothing useful deeper
                return

        self._add_col_read(node.slice)
        if isinstance(node.slice, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
            self._collect_filter(node.slice)
        self.generic_visit(node)

    # ── call analysis ────────────────────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Attribute):
            self.generic_visit(node)
            return

        attr = node.func.attr
        self.all_methods.add(attr)

        # FIX I-05: column-bearing kwargs only go through _add_col_read
        for kwarg_name in _COL_KWARG_MAP.get(attr, []):
            kw_node = _kw(node, kwarg_name)
            if kw_node is not None:
                self._add_col_read(kw_node)
            elif node.args and kwarg_name == _COL_KWARG_MAP[attr][0]:
                self._add_col_read(node.args[0])

        if attr == "rename":
            kw_node = _kw(node, "columns")
            if kw_node and isinstance(kw_node, ast.Dict):
                for k in kw_node.keys:
                    self._add_col_read(k)
                for v in kw_node.values:
                    self._add_col_written(v)

        if attr == "assign":
            for kw in node.keywords:
                if kw.arg:
                    self.cols_written.add(kw.arg)

        if attr in ("merge", "merge_asof", "join"):
            self._collect_join(node, attr)

        if attr == "groupby":
            self.has_groupby = True
            by_node = _kw(node, "by") or (node.args[0] if node.args else None)
            if by_node:
                self._add_col_read(by_node)

        if attr == "agg":
            self._collect_agg(node)

        if attr in _AGG_FUNCS and not isinstance(node.func.value, ast.Name):
            parent = node.func.value
            if isinstance(parent, ast.Call) and \
               isinstance(parent.func, ast.Attribute) and \
               parent.func.attr == "groupby":
                by_node = _kw(parent, "by") or \
                          (parent.args[0] if parent.args else None)
                self.aggregations.append({
                    "groupby_keys":   _str_vals(by_node),
                    "agg_functions":  [attr],
                    "input_columns":  [],
                    "output_columns": [],
                    "raw":            ast.unparse(node)[:200],
                })

        if attr == "rolling":
            self.has_rolling = True
            win = _safe_eval(node.args[0]) if node.args \
                  else _safe_eval(_kw(node, "window"))
            self.window_funcs.append({
                "type": "rolling", "window": win,
                "raw":  ast.unparse(node)[:120],
            })
        if attr == "expanding":
            self.has_window_op = True
            self.window_funcs.append({
                "type": "expanding", "raw": ast.unparse(node)[:120],
            })
        if attr == "shift":
            self.has_window_op = True
            periods = _safe_eval(node.args[0]) if node.args \
                      else _safe_eval(_kw(node, "periods"))
            self.window_funcs.append({
                "type": "lag_lead", "periods": periods,
                "raw":  ast.unparse(node)[:120],
            })
        if attr in ("cumsum", "cumprod", "cummax", "cummin"):
            self.has_window_op = True
            self.window_funcs.append({"type": attr, "raw": ast.unparse(node)[:80]})
        if attr == "rank":
            self.has_window_op = True
            self.window_funcs.append({
                "type":      "rank",
                "method":    _safe_eval(_kw(node, "method")),
                "ascending": _safe_eval(_kw(node, "ascending")),
                "raw":       ast.unparse(node)[:120],
            })

        if attr in ("query", "filter"):
            expr = _safe_eval(node.args[0]) if node.args else None
            self.filters.append({"type": attr, "expression": expr,
                                  "raw": ast.unparse(node)[:100]})
        if attr == "isin":
            vals = _safe_eval(node.args[0]) if node.args else None
            self.filters.append({"type": "isin", "values": vals,
                                  "raw": ast.unparse(node)[:100]})
        if attr == "between":
            self.filters.append({
                "type": "between",
                "low":  _safe_eval(node.args[0]) if len(node.args) > 0 else None,
                "high": _safe_eval(node.args[1]) if len(node.args) > 1 else None,
                "raw":  ast.unparse(node)[:100],
            })
        if attr == "where":
            self.filters.append({"type": "where", "raw": ast.unparse(node)[:100]})
        if attr == "dropna":
            subset_node = _kw(node, "subset")
            if subset_node:
                subset = _str_vals(subset_node)
                self._add_col_read(subset_node)
                self.filters.append({"type": "dropna", "subset": subset,
                                      "raw": ast.unparse(node)[:100]})

        if attr in _STRING_OPS:  self.string_ops.add(attr)
        if attr in _NULL_OPS:    self.null_ops.add(attr)
        if attr in _TYPE_OPS:    self.type_ops.add(attr)

        if attr in _SORT_OPS:
            self.sort_ops.add(attr)
            by_node = _kw(node, "by") or (node.args[0] if node.args else None)
            if by_node:
                self._add_col_read(by_node)

        if attr in _DEDUP_OPS:
            self.dedup_ops.add(attr)
            sub = _kw(node, "subset")
            if sub:
                self._add_col_read(sub)

        if attr in _RESHAPE_OPS:
            self.reshape_ops.add(attr)
            if attr == "explode":
                self.has_explode = True
                col_arg = node.args[0] if node.args else _kw(node, "column")
                if col_arg:
                    self._add_col_read(col_arg)
            if attr in ("pivot_table", "pivot"):
                self.has_pivot = True
                for kname in ("index", "columns", "values"):
                    kn = _kw(node, kname)
                    if kn:
                        self._add_col_read(kn)
            if attr == "melt":
                for kname in ("id_vars", "value_vars"):
                    kn = _kw(node, kname)
                    if kn:
                        self._add_col_read(kn)

        if attr == "apply":
            self.has_apply = True
            axis_val = _safe_eval(_kw(node, "axis"))
            if axis_val == 1:
                self.is_udf_candidate = True
            if node.args:
                fn = node.args[0]
                if isinstance(fn, ast.Lambda):
                    args = fn.args.args
                    if args and args[0].arg in ("row", "r"):
                        self.is_udf_candidate = True

        if attr == "applymap":
            self.has_apply = True
            self.is_udf_candidate = True

        if attr == "transform":
            self.has_window_op = True

        self.generic_visit(node)

    # ── filter collection ────────────────────────────────────────────────

    def _collect_filter(self, node: "ast.expr") -> None:
        if isinstance(node, ast.Compare):
            col = None
            if isinstance(node.left, ast.Subscript):
                strs = _str_vals(node.left.slice)
                col = strs[0] if strs else None
            elif isinstance(node.left, ast.Attribute):
                col = node.left.attr
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
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            self._collect_filter(node.operand)

    # ── join collection ──────────────────────────────────────────────────

    def _collect_join(self, node: ast.Call, join_type: str) -> None:
        if join_type == "merge_asof":
            self.has_merge_asof = True
        info: dict[str, Any] = {
            "join_method": join_type,
            "how":         _safe_eval(_kw(node, "how")) or "inner",
            "on":          None, "left_on": None,
            "right_on":    None, "by":      None,
            "direction":   _safe_eval(_kw(node, "direction")),
            "indicator":   _safe_eval(_kw(node, "indicator")),
            "suffixes":    _safe_eval(_kw(node, "suffixes")),
            "right_table": ast.unparse(node.args[0])[:80] if node.args else None,
            "raw":         ast.unparse(node)[:200],
        }
        if join_type == "merge_asof":
            info["pyspark_note"] = (
                "No direct PySpark equivalent. Use Window.partitionBy(by)"
                ".orderBy(on).rowsBetween(unbounded, currentRow) + "
                "F.last(rate_col, ignorenulls=True)."
            )
        for kname in ("on", "left_on", "right_on", "by"):
            kv = _kw(node, kname)
            if kv:
                strs = _str_vals(kv)
                info[kname] = strs[0] if len(strs) == 1 else (strs or None)
                self._add_col_read(kv)
        # NOTE: indicator= creates a temporary column — do NOT add to cols_read
        self.joins.append(info)


    def _collect_agg(self, node: ast.Call) -> None:
        funcs_used:  set[str] = set()
        input_cols:  set[str] = set()
        output_cols: set[str] = set()
        for kw in node.keywords:
            if kw.arg:
                output_cols.add(kw.arg)
                if isinstance(kw.value, ast.Tuple) and len(kw.value.elts) >= 2:
                    src_strs = _str_vals(kw.value.elts[0])
                    fn_strs  = _str_vals(kw.value.elts[1])
                    input_cols.update(src_strs)
                    funcs_used.update(fn_strs)
                    self._add_col_read(kw.value.elts[0])
        if node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Dict):
                for k, v in zip(arg.keys, arg.values):
                    src_strs = _str_vals(k)
                    fn_strs  = _str_vals(v)
                    input_cols.update(src_strs)
                    funcs_used.update(fn_strs)
                    self._add_col_read(k)
            else:
                funcs_used.update(_str_vals(arg))
        groupby_keys: list[str] = []
        parent_val = node.func.value
        if isinstance(parent_val, ast.Call) and \
           isinstance(parent_val.func, ast.Attribute) and \
           parent_val.func.attr == "groupby":
            by_node = _kw(parent_val, "by") or \
                      (parent_val.args[0] if parent_val.args else None)
            if by_node:
                groupby_keys = _str_vals(by_node)
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

def _classify(func: dict) -> tuple[list[str], str, str, list[str]]:
    ops     = set(func.get("all_methods", []))
    joins   = func.get("joins", [])
    aggs    = func.get("aggregations", [])
    windows = func.get("window_functions", [])
    filters = func.get("filters", [])

    has_merge_asof = func.get("has_merge_asof", False)
    has_rolling    = func.get("has_rolling", False)
    has_window     = func.get("has_window_op", False)
    has_explode    = func.get("has_explode", False)
    has_pivot      = func.get("has_pivot", False)
    is_udf         = func.get("is_udf_candidate", False)
    has_groupby    = func.get("has_groupby", False)
    has_dedup      = bool(func.get("dedup_operations"))
    has_concat     = "concat" in ops or "append" in ops

    types: set[str] = set()
    if func.get("null_operations") or func.get("type_operations") or \
       func.get("string_operations"):
        types.add("cleaning")
    if func.get("type_operations"):
        types.add("type_casting")
    if func.get("string_operations"):
        types.add("string_normalization")
    if joins:
        types.add("asof_join" if has_merge_asof else "join_enrichment")
    if has_groupby and (aggs or any(a in ops for a in _AGG_FUNCS)):
        types.add("aggregation")
    if has_rolling or has_window or windows:
        types.add("window_computation")
    if any(op in ops for op in ("cut", "qcut", "where", "np_where", "select")):
        types.add("conditional_derivation")
    if filters or "dropna" in ops or "isin" in ops or "query" in ops:
        types.add("filtering")
    if has_explode or has_pivot or func.get("reshape_operations"):
        types.add("reshaping")
    if has_dedup:
        types.add("deduplication")
    if has_concat:
        types.add("union")
    if is_udf:
        types.add("custom_transformation")
    if func.get("has_apply") and not is_udf:
        types.add("window_computation")
    if func.get("sort_operations"):
        types.add("sorting")
    if not types:
        types.add("other")

    patterns: set[str] = set()
    risk = "low"
    warnings: list[str] = []

    if has_merge_asof:
        patterns.add("asof_join"); risk = "high"
        warnings.append(
            "pd.merge_asof has no direct PySpark equivalent. "
            "Correct pattern: Window.partitionBy(by).orderBy(on)"
            ".rowsBetween(Window.unboundedPreceding, Window.currentRow)"
            " + F.last(col, ignorenulls=True)."
        )
    if is_udf:
        patterns.add("udf_candidate")
        if risk == "low": risk = "medium"
        warnings.append(
            "apply(axis=1) detected — do NOT use F.udf (Python-serialization-bound). "
            "Rewrite as F.when/otherwise chain or F.pandas_udf for complex logic."
        )
    if has_rolling:
        patterns.add("window_function")
        if risk == "low": risk = "medium"
        warnings.append(
            "Time-based rolling requires rangeBetween(-N*86400, 0) with ts.cast('long'). "
            "Using rowsBetween gives wrong results on sparse data."
        )
    if (has_window or windows) and "window_function" not in patterns:
        patterns.add("window_function")
        if risk == "low": risk = "medium"
    if joins and not has_merge_asof:
        patterns.add("join")
        warnings.append("Broadcast small dimension tables: F.broadcast(dim_df).")
    if has_groupby and aggs:
        patterns.add("aggregation")
    if aggs and not has_groupby:
        patterns.add("two_phase")
    if has_explode:
        patterns.add("explode")
        warnings.append(
            "pandas .explode() drops empty lists (= F.explode). "
            "Use F.explode_outer to retain rows with empty/null arrays."
        )
    if has_concat:
        patterns.add("union")
    if has_pivot:
        patterns.add("aggregation")
    if "quantile" in ops and not has_groupby:
        patterns.add("two_phase")
        if risk == "low": risk = "medium"
    if not patterns:
        patterns.add("pure_map")

    final_pattern = next(iter(patterns)) if len(patterns) == 1 else "mixed"
    if final_pattern == "mixed" and risk == "low":
        risk = "medium"
    return sorted(types), final_pattern, risk, warnings


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
            # sets are not JSON-serializable — convert to sorted list
            if isinstance(val, set):
                val = sorted(val, key=str)
            consts[name] = val
    return consts


def _extract_tag_and_comment(
    line_no: int,
    src_lines: list[str],
    prev_def_line: int,
) -> tuple[str | None, str]:
    """
    FIX I-11: scan only from prev_def_line (exclusive) to line_no (exclusive).
    This prevents tags from the previous function leaking into the next one.
    """
    tag, pyspark = None, ""
    start = max(0, prev_def_line)        # line after previous def (0-indexed)
    end   = max(0, line_no - 1)          # line before current def
    block = src_lines[start:end]
    for line in block:
        s = line.strip()
        m = re.search(r"\[(T\d{2,3})\]", s)
        if m:
            tag = m.group(1)
        if "PySpark" in s and ("equivalent" in s or "PySpark:" in s):
            pyspark = re.sub(r"^#\s*", "", s).strip()
    return tag, pyspark


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
    """
    FIX I-08: also include calls made in the __main__ guard block,
    stored under the key "__main__".
    """
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
    """
    Resolve a local import to a .py file.  Returns (resolved_path, tried_paths).
    resolved_path is None when not found; tried_paths lists everything checked
    so callers can emit a useful warning (FIX I-01).

    Search strategy (in order):
      1. base_dir  (pipeline's own directory)
      2. base_dir.parent   (project root if pipeline lives one level down)
      3. base_dir.parent.parent  (grandparent, for deeper nesting)
      4. If base_dir.name matches the top-level package name, also search
         base_dir.parent (handles "pipeline lives inside the package" case)

    For each root we try:
      a. <root>/<dotted/path>.py           (package.module -> package/module.py)
      b. <root>/<dotted/path>/__init__.py  (sub-package)
      c. <root>/<basename>.py              (flat sibling: generate_synthetic_data.py)
    """
    base_dir = Path(base_dir)

    if not module_name and level == 0:
        return None, []

    # Build search_dir for relative imports (level > 0)
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
    """
    Parse a Python data-transformation pipeline and return a comprehensive
    JSON-serialisable analysis dict.
    """
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

    # FIX I-11: build a map of def-line -> previous-def-end-line so the tag
    # scanner never reaches into the previous function's body.
    def_lines = sorted(
        n.lineno for n in tree.body if isinstance(n, ast.FunctionDef)
    )
    # end_line of each top-level def (0-indexed into src_lines)
    end_line_map: dict[int, int] = {}
    top_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for i, fn in enumerate(top_defs):
        prev_end = top_defs[i - 1].end_lineno if i > 0 else 0
        end_line_map[fn.lineno] = prev_end  # previous function's last line (1-indexed)

    functions: list[dict] = []
    for func_node in tree.body:
        if not isinstance(func_node, ast.FunctionDef):
            continue
        if func_node.name.startswith("__"):
            continue

        # FIX I-11: pass previous-def end line to bound the tag search window
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
        "unresolved_imports":  [],   # FIX I-01
    }

    # ── follow local imports ─────────────────────────────────────────────
    if follow_imports:
        base_dir = path.parent
        for imp in imports.get("local", []):
            module_name = imp.get("module") or ""
            level       = imp.get("level", 0)

            dep_path, tried = _resolve_local_import(module_name, base_dir, level)

            # FIX I-01: warn loudly on miss instead of silently continuing
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
# CLI
# ---------------------------------------------------------------------------

def _wrap(items: list[str], indent: str = "      ", width: int = 100) -> str:
    """
    Join items onto wrapped lines.  Breaks only BETWEEN items (after a comma),
    never mid-identifier.  Measures the actual rendered line to guarantee
    no overflow.  A single item longer than `width` is kept on its own line.
    """
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
    """
    Print a full summary block for one parsed module (main file or dependency).
    Uses a vertical per-function layout so nothing is ever truncated.
    """
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

    # ── per-function blocks ───────────────────────────────────────────────
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

        # Output cols: prefer new columns; fall back to modified (prefixed ~)
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

        # warnings
        for w in f["pyspark"].get("warnings", []):
            print(f"      ⚠  {w}", file=out)

    # ── all columns referenced ────────────────────────────────────────────
    all_cols = s.get("all_columns_referenced", [])
    if all_cols:
        print(f"\n  All columns referenced ({len(all_cols)}):", file=out)
        print(_wrap(all_cols, indent="    ", width=100), file=out)


def _print_summary(r: dict, out: Any = None) -> None:
    """
    Top-level summary printer.
    Renders the main file as a full block, then renders each dependency
    as its own full block with the same detail level.

    FIX (truncation):   uses _wrap() — no fixed-width columns, nothing
                        is ever cut with '...'.
    FIX (dependencies): each dependency gets _print_module_summary(),
                        identical in detail to the main file rendering.
    """
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

def run_parser(script_path: str, follow_imports: bool = False, output_dir: str | None = None) -> None:
    print(f"Parsing pipeline: {script_path} with follow_imports={follow_imports}", file=sys.stderr)
    result = parse_pipeline(script_path, follow_imports=follow_imports)

    buf = io.StringIO()
    _print_summary(result, out=buf)
    summary_text = buf.getvalue()
    sys.stderr.write(summary_text)
    sys.stderr.flush()

    input_path = Path(script_path)
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = input_path.parent / f"{input_path.stem}_parsed"

    output_dir.mkdir(parents=True, exist_ok=True)

    json_file    = output_dir / f"{input_path.stem}_parsed.json"
    summary_file = output_dir / f"{input_path.stem}_summary.txt"

    try:
        json_str = json.dumps(result, indent=2)
    except TypeError as exc:
        print(f"[ERROR] Result is not JSON-serializable: {exc}", file=sys.stderr)
        sys.exit(1)

    json_file.write_text(json_str, encoding="utf-8")
    summary_file.write_text(summary_text, encoding="utf-8")

    return {"json_file": str(json_file), "summary_file": str(summary_file)}


if __name__ == "__main__":
    import argparse
    import io

    ap = argparse.ArgumentParser(
        description="Comprehensive AST parser for Python data-transformation pipelines."
    )
    ap.add_argument("script", help="Path to the Python pipeline file to parse.")
    
    ap.add_argument(
        "--follow-imports", action="store_true", default=False,
        help="Also parse every resolvable local dependency.",
    )

    ap.add_argument(
        "--output", default=None, metavar="DIR",
        help=(
            "Directory to write JSON + summary files. "
            "Defaults to <script_stem>_parsed/ next to the input file."
        ),
    )
    args = ap.parse_args()
    run_parser(args.script, follow_imports=args.follow_imports, output_dir=args.output)
    