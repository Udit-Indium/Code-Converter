# PySpark Visualisation Conventions

How to convert any **plotting / charting / reporting** code (matplotlib,
seaborn, pandas `.plot`, `df.hist`, etc.) found in the source script. Most ETL
scripts have little or no visualisation; if the source has none, this file does
not apply — **do not invent charts**.

The golden rule: **plotting libraries are driver-side and work on small,
collected data.** Spark distributes the *computation*; the *plot* is always
produced on the driver from an aggregated, deliberately small result.

---

## 1. The aggregate-then-collect pattern

Never plot from a raw distributed DataFrame, and never `toPandas()` a large one.
Do the heavy aggregation in Spark, reduce to a small summary, then collect that
summary for the plotting library.

✅ **Follow:**
```python
def plot_monthly_revenue(enriched: DataFrame) -> "pd.DataFrame":
    """Aggregate revenue per month in Spark, then return a SMALL pandas frame
    ready for plotting. Converted from `plot_monthly_revenue`."""
    monthly = (
        enriched.groupBy("month")                      # heavy work stays distributed
                .agg(F.sum("amount_usd").alias("revenue_usd"))
                .orderBy("month")
    )
    pdf = monthly.toPandas()                            # SMALL: one row per month
    # caller renders: pdf.plot(x="month", y="revenue_usd", kind="bar")
    return pdf
```
❌ **Avoid:** collecting the full frame to the driver before aggregating —
defeats distribution and can OOM the driver:
```python
pdf = enriched.toPandas()                              # ❌ entire dataset on driver
pdf.groupby("month")["amount_usd"].sum().plot.bar()
```

---

## 2. Keep the plotting call faithful; separate compute from render

- The Spark function returns the **small aggregated pandas frame**; the actual
  `matplotlib`/`seaborn` call stays as in the source (it is driver-side Python
  and needs no translation).
- Preserve the source's chart type, axes, labels, bins, and ordering exactly —
  visualisation output must match.
- `df.hist()` / `qcut`-based bucketing → aggregate counts per bucket in Spark
  (`F.when` bucketing or `F.percentile_approx` edges, flagged as approximate per
  ETL conventions §5), then plot the small bucket table.

---

## 3. Databricks `display()`

- In Databricks notebooks, prefer `display(df)` over `df.show()` for tabular/
  visual output — it renders natively and can chart without a driver collect.
- For a `.py` module (not a notebook) keep a `df.show(n)` or return the frame so
  the module stays runnable in the local sandbox; note `display()` as the
  in-Databricks equivalent in a comment.

✅ `display(monthly)  # Databricks-native chart; df.show(20) in the local sandbox`

---

## 4. SEMS still applies to plotting code

- Docstring on every plotting function (what it charts, source function).
- No hardcoded output paths for saved figures (`plt.savefig("/Users/me/…")` ❌);
  take the path as a parameter / config.
- `logging`, not `print`, for status.

---

## Visualisation checklist

- [ ] Source has visualisation? If **no**, none added. If **yes**, every chart converted.
- [ ] Heavy aggregation done in Spark; only a **small** summary collected to the driver.
- [ ] No `.toPandas()` / `.collect()` on a large/raw frame for plotting.
- [ ] Chart type, axes, labels, bins, ordering preserved exactly.
- [ ] `display()` noted for Databricks; module stays sandbox-runnable.
- [ ] SEMS rules (docstring, no hardcoded paths, logging) applied to plotting code too.
