# PySpark ETL Conversion Conventions

The rules every converted PySpark module **must** follow when translating a
pandas / Python ETL script to distributed PySpark. They are derived from the
**BRD (Python → PySpark Code Conversion for the Databricks STAM layer)** and from
the known *silent-divergence traps* of pandas → Spark.

The converter is **generic** — these rules apply to **any** source script + its
`*_parsed.json` (AST analysis). Never hardcode a specific pipeline, column, or
function name.

Two inputs, three jobs:
| Input | Where | Role |
|---|---|---|
| the parsed AST inventory | on disk — surfaced by `read_source_index_tool` | **Contract of WHAT to convert** — constants, the function/class inventory + per-function Spark hints. Its `call_graph` is also the **contract of the FLOW**: the order the orchestrator calls the functions in. |
| the original Python source | on disk — read it with `read_source_index_tool` / `read_source_functions_tool` | **Source of truth for behaviour.** When the parsed JSON and the source disagree, trust the source and note the gap. |

---

## 0. The four hard rules (non-negotiable)

1. **Convert EVERY constants, function and EVERY class — no exceptions, no placeholders.**
   The converted constant counts and callable count must **equal** the parsed-JSON inventory. 
   No "representative subset", no `# TODO: convert later`,
   no stubbed bodies, no silent drops. A thin validator, a nested helper, or the
   orchestrator (`run_pipeline`) is still converted. If a function is genuinely
   un-translatable as-is, convert it to the closest faithful PySpark form and
   record it under **manual interventions** — never omit it.
2. **The orchestrator's flow must match the source's call order** . Same
   number of constants along with their assigned values, functions, same order, 
   same data hand-off between steps.
3. **Preserve behaviour exactly (functional equivalence).** Same inputs → same
   outputs as the original. Do not add, drop, reorder, or "improve" business
   logic. The converted code is a *translation*, not a rewrite.
4. **Distributed DataFrame API, driven by the inputs.** Use the DataFrame API
   and column expressions — no row loops, no `.collect()`/`.toPandas()` inside a
   transform, **no plain `F.udf`** unless there is truly no built-in. Every
   column name, join key, group key, filter literal and window spec comes from
   the **parsed JSON + source, used verbatim** — never invented or renamed.

---

## 1. How to count functions from the parsed AST JSON (Requirement 1)

The parsed inventory is the list you must exhaust. The count of functions and 
total_columns_referenced are given in the summary  :
fields given in its `summary` field are defined below:
"function_count",
"total_columns_referenced",
"all_columns_referenced",
"transformation_type_distribution",
"pyspark_risk_distribution",
"all_join_types",
"all_aggregation_functions",
"udf_candidates",
"high_risk_functions",
"dependencies_parsed",
"dependencies_function_count",
"dependencies_high_risk",
"dependencies_udf_candidates"

all the dependencies are defined in this field of the inventory:
"dependencies"

apart from this, every referenced file and its path is also given in the inventory.

- `parsed["functions"]` lists **top-level** functions only. Take the number from
  `read_source_index_tool()` for the script in front of you — never from an
  example. Counts vary hugely: a notebook restructured by the refactor stage
  routinely has 60+.
- `parsed["classes"]` lists classes; each has a `methods` list. Convert the
  class and every method. (`PipelineError(Exception)` has no methods → it is
  carried over verbatim as an exception class.)
- `parsed["constants"]` lists every constant with its value from the source script.
- **Nested helper functions** (a `def _fill(...)` *inside* `impute_nulls`) do
  **NOT** appear in `parsed["functions"]`. They live in the parent's body and
  are converted **as part of converting the parent**. You can confirm they exist
  by reading the parent with `read_source_functions_tool`.

> ⚠️ **Reconciling the two inputs:** the source contains **all** functions —
> top-level **and** nested — so a count taken from the source is **higher** than
> `len(parsed["functions"])` — e.g. a script with 5 nested helpers reports 5
> fewer than the source contains. This is expected. Reconcile the
> parsed count against the graph's **top-level** functions, and make sure every
> nested helper from the graph still exists inside its parent in your output.

**Self-check before you finish:** call `read_converted_file_tool()` and compare
its `function_names` against `read_source_index_tool()`. You have no shell, so
do not try to `grep`. List any divergence in the report; never let the count
silently drop.

✅ **Positive:** every top-level function from the index is present, nested
helpers still inside their parents, exception classes carried over → 1:1 with
the inventory.

❌ **Negative:** "I converted the 12 most important transforms and stubbed the
rest with `pass # TODO`." — violates rule 0.1; the run is a failure.

---

## 2. How to reproduce the flow (Requirement 2)

The flow comes from **`parsed["call_graph"]`** — a mapping of each function to the
functions it calls, in call order. To recover the pipeline order deterministically:

1. The **orchestrator** is the function with the most entries in its `call_graph`
   list. A refactored notebook usually names it `main`.
2. Its flow is that list, in order. That ordered sequence of callee names is
   exactly what the converted orchestrator must execute.
3. Cross-check against the source: read the orchestrator itself with
   **read_source_functions_tool(function_names=["run_pipeline"])**. If the call
   graph and the source differ, trust the **source**.

**Repeated calls are real.** If `validate_schema` appears 3x in the flow (once
per input frame), call it 3x in the converted orchestrator — do **not** dedupe.

The **data hand-off** — which frame feeds which step — is read off the
orchestrator's body: each call's arguments say what its input is, and what its
result is bound to. Wire each step's input to the correct prior result, exactly
as the source does (e.g. `transactions_ordered` flows into
`aggregate_customer_metrics`, not the raw `transactions`). This is the one thing
the call graph alone cannot tell you, so read the orchestrator before writing it.

✅ **Positive:** the converted `run_pipeline` calls the same functions in the same
order, threading the same intermediate frames, as the source does.

❌ **Negative:** reordering "because Spark can do it lazily anyway", or collapsing
the 3 `validate_schema` calls into 1 — changes observable behaviour/flow.

---

## 3. Module skeleton (Databricks STAM-ready, locally runnable)

```python
"""
<stem>_spark.py — PySpark conversion of <stem>.py
Target: Databricks STAM layer, PySpark 3.5+ (DataFrame API).
Flow and functions mirror the parsed AST inventory.
"""
from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import types as T

logger = logging.getLogger(__name__)

# CONFIG carried over VERBATIM from the source constants (parsed JSON -> constants).
CONFIG = { ... }   # same keys/values as the source — do not re-tune


def get_spark(app_name: str = "py2snow") -> SparkSession:
    """On Databricks the `spark` session already exists; this also makes the
    module runnable locally (sandbox)."""
    return SparkSession.builder.appName(app_name).getOrCreate()
```

- **Requirement 4 — a proper SparkSession.** The module must expose a
  `get_spark()` (or accept an injected `spark`), so it runs both on a cluster and
  in a local sandbox. Set the session time zone to UTC so tz-aware datetimes
  line up with pandas.
- **Signatures:** `def f(df: pd.DataFrame, ...) -> pd.DataFrame` becomes
  `def f(df: DataFrame, ...) -> DataFrame`. Keep the **same function name,
  parameter names, and order** (traceability).
- **Spark is immutable & lazy:** build with `withColumn` / `select` / `join` /
  `groupBy().agg()`; never mutate in place. Return the new `DataFrame`.
- **No I/O in transforms:** keep read/write at the edges (STAM patterns);
  transform functions take and return DataFrames.

### 3a. Imports discipline (do NOT miss an import)

Every name you use MUST be imported. The file is assembled batch-by-batch, so
after writing each batch, re-check every non-local name you referenced and make
sure its import line is present. A missing import is one of the most common
failures (`NameError: name 'X' is not defined` at test time).

**Always include (baseline for every converted module):**
```python
from __future__ import annotations
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
```

**Name → import lookup — if you use the name on the left, include the import on the right:**

| If you use…                                   | Add this import |
|-----------------------------------------------|-----------------|
| `Window`, `WindowSpec`                        | `from pyspark.sql.window import Window` (`, WindowSpec`) |
| `StructType`,`StructField`,`StringType`,`IntegerType`,`LongType`,`DoubleType`,`FloatType`,`BooleanType`,`TimestampType`,`DateType`,`ArrayType`,`MapType`,`DecimalType` | `from pyspark.sql.types import <Name>` (or use `T.<Name>`) |
| `Bucketizer`,`StringIndexer`,`OneHotEncoder`  | `from pyspark.ml.feature import <Name>` |
| `chain`                                       | `from itertools import chain` |
| `reduce`                                      | `from functools import reduce` |
| `datetime`,`timedelta`,`date`                 | `from datetime import datetime, timedelta, date` |
| `Decimal`                                     | `from decimal import Decimal` |
| `defaultdict`,`Counter`                       | `from collections import defaultdict, Counter` |
| `math`,`re`,`random`,`json`,`functools`,`itertools` | `import <module>` |

**Rule:** prefer the namespaced forms `F.<fn>` and `T.<Type>` (only the two
imports above needed) over importing dozens of individual functions/types. Put
all imports ONCE at the top of the file (the assembler dedupes them); include
any import a batch needs in that same batch.

---

## 4. pandas → PySpark mapping (drive off `transformation_types` / `pyspark.pattern`)

For each function read its parsed `pyspark.pattern`, `pyspark.risk`,
`pyspark.warnings`, and the construct lists (`joins`, `aggregations`, `filters`,
`window_functions`), then apply the matching row.

| Parsed pattern / construct | PySpark translation |
|---|---|
| `pure_map` — `df["c"]=expr`, `astype`, `str.*`, `fillna`, `np.where` | `withColumn("c", <expr>)`; `.cast()`; `F.lower/upper/trim/regexp_replace`; `F.coalesce`/`.fillna`; `F.when(...).otherwise(...)`. |
| `string_normalization` | `F.trim`, `F.lower`, `F.upper`, `F.initcap`, `F.regexp_replace`, `F.regexp_extract`, `F.split`. pandas `.str.lower()` == `F.lower`. |
| `type_casting` | `.cast(T.X())`; dates via `F.to_timestamp`/`F.to_date` with an **explicit format**; pandas `errors="coerce"` → bad parses become `null` (matches `to_timestamp`). |
| `filtering` (`filters[]`) | `.filter(<predicate>)`/`.where(...)`; `isin` → `F.col(c).isin(...)`; `between` → `.between(lo,hi)`; `dropna(subset=)` → `.dropna(subset=[...])`. Use the **exact** `filters[].op`/`value`. |
| `join_enrichment` (`joins[]`) | `left.join(F.broadcast(right), on, how)` — `how`/keys from `joins[].how`/`on`/`left_on`/`right_on`. **Broadcast** small dimension tables. |
| `asof_join` (`has_merge_asof`, risk **high**) | **No native equivalent.** `Window.partitionBy(by).orderBy(on).rowsBetween(Window.unboundedPreceding, Window.currentRow)` + `F.last(value, ignorenulls=True)` after a `<=` filter. A naive join+filter returns ALL prior matches → wrong values. |
| `aggregation` (`aggregations[]`) | `df.groupBy(*keys).agg(F.<fn>(input).alias(output))` — keys/fns/outputs from `aggregations[]`. pandas `nunique`→`F.countDistinct`; `mean`→`F.avg`; `std`→`F.stddev` (sample). |
| `window_computation` (`window_functions[]`) | `Window.partitionBy(...).orderBy(...)` + `F.lag/lead/rank/dense_rank/row_number/sum/avg().over(w)`. `shift(n)`→`F.lag(col,n)`. cumsum→`F.sum().over(w.rowsBetween(unboundedPreceding,currentRow))`. |
| rolling (risk medium) | **Time-based** rolling needs `rangeBetween(-N*86400, 0)` over `ts.cast("long")` — NOT `rowsBetween` (wrong on irregular timestamps). Count-based → `rowsBetween(-(N-1), 0)`. |
| `conditional_derivation` (`pd.cut`/`qcut`/`np.select`/`np.where`) | chained `F.when(...).when(...).otherwise(...)`. `pd.cut`: mind **right- vs left-inclusive** (replicate `right=`/`include_lowest=`). `qcut`→`F.percentile_approx` (approximate — flag divergence). |
| `reshaping` — `explode` | `F.explode` **drops empty/null arrays**; use `F.explode_outer` to keep those rows. Match the source exactly. |
| `reshaping` — `pivot_table`/`crosstab` | `groupBy(index).pivot(column).agg(F.<fn>(value))`. Fill missing cells to match pandas `fill_value`. |
| `deduplication` | `drop_duplicates(subset)`→`.dropDuplicates([...])`; for "keep last after sort" use `row_number().over(Window.partitionBy(keys).orderBy(order.desc()))==1` (plain `dropDuplicates` keeps an arbitrary row). |
| `union` (`pd.concat`) | `a.unionByName(b, allowMissingColumns=True)` — **name-based** like pandas `concat`, not positional `union()`. |
| `custom_transformation`/`is_udf_candidate` (`apply(axis=1)`) | **Rewrite as column expressions first** (`F.when` chains, arithmetic, `F.sha2`, `F.concat_ws`). Only if irreducibly Python, use a typed `@F.pandas_udf` (vectorized) — **never** plain `F.udf`. |

> When `pyspark.pattern == "mixed"`, decompose the function into its constituent
> steps and apply the matching row to each, in source order.

✅ **Positive — an aggregation translated from the parsed hints:**
```python
# parsed: aggregations=[{groupby:["customer_id"], fn:"nunique", input:"product_id",
#                        output:"distinct_product_count"}]
out = enriched.groupBy("customer_id").agg(
    F.countDistinct("product_id").alias("distinct_product_count")  # nunique -> countDistinct
)
```
❌ **Negative — pulling to the driver / row loop:**
```python
pdf = enriched.toPandas()                 # ❌ defeats distribution
counts = {}
for row in pdf.itertuples():              # ❌ row loop
    counts[row.customer_id] = ...
```

---

## 5. Silent-divergence traps (produce WRONG output, not errors)

Handle each explicitly; these pass a smoke test but corrupt results:

- **as-of join** (`merge_asof`): nearest-prior semantics — Window + `F.last(ignorenulls=True)`, never a plain range join.
- **null ordering:** Spark sorts `null` **first** ascending by default; pandas `sort_values` puts `NaN` **last**. Use `F.col(c).asc_nulls_last()`/`desc_nulls_last()` to match.
- **`count` vs nulls:** `F.count(col)` skips nulls (like pandas); `F.count("*")` counts rows. `nunique` excludes NaN → `F.countDistinct` (also excludes null). Pick to match.
- **integer/float division:** pandas may upcast; Spark integer division truncates. `.cast("double")` where the source produces floats.
- **string trimming/case:** replicate the source's exact order (`strip`→`lower`→`replace`); `F.trim` removes only spaces — use `F.regexp_replace` for other whitespace.
- **`explode` empty/null:** `F.explode` drops, `F.explode_outer` keeps — match the source.
- **timestamps & timezone:** parse with an explicit format and a fixed session tz (`spark.sql.session.timeZone="UTC"`).
- **`percentile_approx`/`qcut`/`median`:** approximate in Spark — exact equality fails; widen tolerance for these columns and **note the expected divergence**.
- **`dropDuplicates` non-determinism:** keeps an arbitrary duplicate; if the source keeps first/last, use the `row_number` window pattern.
- **dynamic / f-string columns** (often absent from `columns.new` because the parser can't resolve them): confirm them in the **source** and convert them; do not skip because the parsed JSON omits them.

---

## 6. Final ETL checklist (must all hold)

- [ ] Converted callable count **==** `len(functions) + class methods` from the inventory, **plus `get_spark`** — which §3 requires and the source does not have. It is the ONE permitted addition; everything else must trace to a source function.
- [ ] EXACTLY the source functions are present — **no missing** function (including the orchestrator, e.g. `main`, which is a normal function to convert like any other) and **no invented** helpers beyond `get_spark` (in particular, do not also add a `get_spark_session` — one session helper, named `get_spark`).
- [ ] Every non-local name used has its import (see §3a); no `NameError` waiting at test time.
- [ ] Orchestrator flow **==** the source's call order (same order, repeated calls kept, frames threaded as the original orchestrator threads them).
- [ ] Every column / key / literal / window spec taken **verbatim** from parsed JSON + source.
- [ ] Distributed DataFrame API throughout; no row loops, no driver collects, no plain `F.udf`.
- [ ] Each high-risk construct (`merge_asof`, rolling, `qcut`, dedup-keep, explode-empty) handled per §4–§5.
- [ ] Module exposes a proper `get_spark()`/`SparkSession`; UTC session tz; no I/O inside transforms.
- [ ] SEMS compliance per `SEMS_Compliance.md`; any plotting per `pyspark_visualisations_conventions.md`.
