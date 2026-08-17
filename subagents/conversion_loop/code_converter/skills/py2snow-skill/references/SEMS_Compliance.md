# SEMS Compliance Conventions

**SEMS** = Software Engineering & Maintainability Standards (BRD FR-03 / FR-04).
Every converted PySpark file must be **readable, traceable, secure, and
maintainable**. These are the code-quality rules the converted output is graded
on, independent of whether the logic is correct (that is the ETL conventions'
job).

A converted file passes SEMS only if **all** of the following hold. Each rule has
a ✅ to follow and a ❌ to avoid.

---

## 1. Docstrings — every function and class

- **Every** function (top-level and nested) and **every** class has a docstring.
- The docstring states: what it does, its parameters, what it returns, and — for
  traceability — the **source function it was converted from** and any
  divergence (e.g. "uses `percentile_approx`, approximate vs pandas `qcut`").
- If the source function already had a docstring (see `parsed["functions"][].docstring`),
  **carry its intent over** and extend it; do not drop it.

✅ **Follow:**
```python
def convert_to_usd(transactions: DataFrame, fx_rates: DataFrame) -> DataFrame:
    """Convert each transaction amount to USD using the latest prior FX rate.

    Converted from `convert_to_usd` in customer_analytics_pipeline.py. The pandas
    `merge_asof` (nearest prior rate) is reproduced with a window + F.last over
    rates filtered to on-or-before the txn date — see ETL conventions §5.

    Args:
        transactions: txn frame with `txn_timestamp`, `currency`, `amount`.
        fx_rates:      rates frame with `currency`, `effective_date`, `rate_to_usd`.
    Returns:
        `transactions` plus an `amount_usd` column.
    """
```
❌ **Avoid:** no docstring, or a useless one:
```python
def convert_to_usd(transactions, fx_rates):
    # converts to usd
    ...
```

---

## 2. Inline comments + source mapping (traceability)

- One comment per **non-trivial** transformation explaining *what & why*.
- Carry the source's block tags through. If the source has `# [T07] convert to USD`,
  the converted block keeps `# [T07] convert to USD -> window + F.last`. If the
  parsed JSON gives a function `tag`, use it.
- Comment the **divergence traps** explicitly at the site (null ordering,
  explode_outer, percentile_approx, etc.).
- Do **not** over-comment trivial lines (`# create spark session`) — comment
  intent, not syntax.

✅ **Follow:**
```python
# [T12] flag velocity fraud: >=3 txns within a 60s window per customer
w = Window.partitionBy("customer_id").orderBy(F.col("txn_ts").cast("long"))
out = out.withColumn(
    "has_velocity_flag",
    F.count("*").over(w.rangeBetween(-CONFIG["fraud_velocity_window_seconds"], 0))
     >= CONFIG["fraud_velocity_count_threshold"],
)
```
❌ **Avoid:** unexplained magic, no source mapping:
```python
out = out.withColumn("has_velocity_flag", F.count("*").over(w.rangeBetween(-60, 0)) >= 3)
```
(hardcoded literals instead of `CONFIG`, no `[Txx]` tag, no explanation.)

---

## 3. Naming standards

- **Keep the source's function, parameter, and output-column names verbatim** —
  this is what makes the conversion auditable. Do not rename `out` to `result_df`
  or `customer_id` to `cust_id`.
- Module-level constants stay `UPPER_SNAKE_CASE` (`CONFIG`, `REQUIRED_TXN_COLS`).
- Functions/variables stay `lower_snake_case`. Classes stay `PascalCase`.
- No single-letter names except conventional loop/lambda bounds; mirror whatever
  the source used so columns trace 1:1.

✅ **Follow:** `def normalize_strings(customers: DataFrame) -> DataFrame:` with an
output column `customer_hash` exactly as the source named it.

❌ **Avoid:** `def norm(c):` returning `cust_hash` — breaks traceability and the
column-name contract from the parsed JSON.

---

## 4. Logging, not print

- Use the `logging` module (`logger = logging.getLogger(__name__)`); log at
  block/function boundaries (`logger.info("…")`, `logger.warning("…")`).
- **Never** use `print` for pipeline logging. (The source's `logger.info`/
  `logger.warning` calls map straight across.)
- No noisy per-row logging — that forces materialization and kills performance.

✅ `logger.warning("Found %d unknown currencies; coercing to NaN", n_unknown)`
❌ `print("done converting")`

---

## 5. Error handling & try/except discipline

- **Wrap operations that can genuinely fail at runtime** in `try/except`: external
  reads/writes, `spark.createDataFrame` on synthetic data, type casts that can
  raise, schema validation, and any external lookup/broadcast. Pure lazy DataFrame
  transforms do not each need a try/except (the exception surfaces at the action).
- **Catch SPECIFIC exceptions**, never a bare `except:` and never a blanket
  `except Exception` that hides the cause. Prefer the narrowest type
  (`AnalysisException`, `ValueError`, `KeyError`, `Py4JJavaError`, a domain
  `PipelineError`).
- **Never swallow**: an `except` block must do one of — re-raise, raise a
  domain exception `from exc`, or log at `error`/`warning` and take a defined
  fallback. No `except: pass`, no empty body, no silent `return None`.
- **Preserve the cause**: use `raise PipelineError(...) from exc` so the original
  traceback is retained.
- **Log with context** inside the handler (which function, which input), then act.
- **Raise the same exception types the source raises, where it raised them**
  (e.g. a `PipelineError` from `validate_schema`). Carry custom exception classes
  over verbatim.
- Use `finally` (or a context manager) for cleanup that must always run.

✅ **Follow — validate-and-raise (guard clause):**
```python
missing = set(required_cols) - set(df.columns)
if missing:
    raise PipelineError(f"{name} missing required columns: {sorted(missing)}")
```
✅ **Follow — specific catch, log, re-raise preserving cause:**
```python
try:
    df = spark.read.parquet(path)
except AnalysisException as exc:
    logger.error("Failed to read input for %s at %s: %s", name, path, exc)
    raise PipelineError(f"could not read {name} from {path}") from exc
```
❌ **Avoid — swallowing / broad catch (SonarQube: python:S5754 / S1181):**
```python
try:
    validate(df)
except:            # bare except — forbidden
    pass           # swallows data-quality failures the source surfaces
```
❌ **Avoid — blanket catch that hides the real error:**
```python
try:
    return build(df)
except Exception:
    return None     # silent failure; caller can't tell what broke
```

---

## 6. Security — no hardcoded secrets

- No hardcoded credentials, tokens, absolute secret paths, JDBC passwords, or
  connection strings anywhere in the converted file.
- Configuration stays in `CONFIG` / parameters; secrets come from the runtime
  (e.g. Databricks secret scopes / widgets), never inlined.

❌ **Avoid:** `spark.read.jdbc(url="jdbc:...;password=Pa55w0rd", ...)`
✅ **Follow:** read connection details from injected config/secret references.

---

## 7. Modularity & structure

- One function = one transformation, mirroring the source's boundaries. No
  mega-functions that fuse several source functions together.
- Keep nested helpers nested (do not hoist `_fill` out of `impute_nulls`) — the
  structure must mirror the source so the original call order still maps.
- Imports at the top, grouped (stdlib, third-party, local). `CONFIG` and module
  constants near the top.

---

## 8. SonarQube-style static-quality rules (clean code)

The output is scanned by a SonarQube-style linter. Each converted function must
pass these (the SonarQube rule id is noted for reference):

- **Type hints (python:S...):** every parameter and the return are annotated —
  `def f(df: DataFrame, threshold: float = 0.0) -> DataFrame:`.
- **No commented-out code (python:S125):** never leave dead/commented-out code
  (`# out = df.filter(...)`). Delete it; comments explain *why*, not old code.
- **No dead / unreachable code (python:S1763):** no code after a `return`/`raise`;
  no `if False:` blocks; no unused branches.
- **No unused imports or variables (python:S1128 / python:S1481):** import only
  what you use; don't assign a variable you never read.
- **No leftover TODO/FIXME (python:S1135) or stubs:** no `# TODO`, `...`, `pass`
  placeholder bodies, or `# continue similarly`. Every function is fully
  implemented.
- **No magic literals (python:S109 / S1192):** business numbers/strings come from
  `CONFIG` or named constants, not inlined (`CONFIG["threshold"]`, not `250.0`).
  A repeated string literal (≥3×) becomes a named constant.
- **Manageable complexity (python:S3776 cognitive complexity):** keep functions
  focused; extract a nested helper rather than deeply nesting `when`/`if` chains.
  Avoid functions longer than ~60 lines or nesting deeper than 3 levels.
- **No bare/broad except (python:S5754 / S1181):** see §5.
- **Boolean/return clarity (python:S1125):** no `== True`, no redundant
  `if x: return True else: return False` — return the expression.
- **Consistent returns (python:S3516):** a function that returns a `DataFrame`
  returns one on every path (no accidental `None`).

✅ **Follow:**
```python
HIGH_VALUE_THRESHOLD = CONFIG["high_value_threshold"]  # named, from CONFIG

def filter_high_value_orders(df: DataFrame, threshold: float = HIGH_VALUE_THRESHOLD) -> DataFrame:
    """Keep non-null orders above `threshold`. Converted from filter_high_value_orders."""
    return df.filter((F.col("amount") > threshold) & F.col("amount").isNotNull())
```
❌ **Avoid:** untyped, magic literal, dead comment, `== True`:
```python
def filter_high_value_orders(df, threshold=250.0):
    # df = df.dropna()   # <- commented-out dead code
    return df.filter((F.col("amount") > threshold) == True)
```

---

## SEMS checklist (must all hold)

- [ ] Every function & class has a meaningful docstring naming its source function.
- [ ] Non-trivial transforms commented; source `[Txx]`/tags carried through; divergence traps commented.
- [ ] Function / parameter / column / class names kept **verbatim** from the source + parsed JSON.
- [ ] Type hints on every parameter and return.
- [ ] `logging` used throughout; **no** `print`; no per-row logging.
- [ ] Risky ops wrapped in `try/except` with a SPECIFIC exception, logged, and re-raised (`from exc`) or a defined fallback; **no** bare/broad `except`, **no** `except: pass`.
- [ ] Same exception types raised at the same points as the source.
- [ ] **No** hardcoded secrets / credentials / connection strings.
- [ ] **No** commented-out code, dead code, unused imports/vars, TODO/FIXME, or placeholder bodies.
- [ ] **No** magic literals — business values come from `CONFIG` / named constants.
- [ ] Functions are focused (cognitive complexity, length, nesting kept low).
- [ ] Modular structure mirrors the source; nested helpers stay nested; imports grouped.
