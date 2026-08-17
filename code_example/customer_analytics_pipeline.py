from __future__ import annotations
import hashlib
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pipeline2 import deduplicate_latest, union_with_historical

CONFIG: Dict[str, Any] = {
    "session_gap_minutes": 30,
    "high_value_percentile": 0.95,
    "rolling_window_days": 7,
    "rfm_quantiles": [0.25, 0.50, 0.75],
    "outlier_iqr_multiplier": 1.5,
    "min_transactions_for_active": 3,
    "fraud_amount_threshold_usd": 10_000,
    "fraud_velocity_window_seconds": 60,
    "fraud_velocity_count_threshold": 3,
    "top_n_products_per_customer": 3,
    "cohort_observation_months": 12,
    "tier_thresholds_usd": {"bronze": 0, "silver": 1_000, "gold": 5_000, "platinum": 20_000},
    "age_bins": [-np.inf, 18, 25, 35, 50, 65, np.inf],
    "age_labels": ["minor", "young_adult", "adult", "middle_age", "senior", "elderly"],
    "valid_currencies": {"USD", "EUR", "GBP", "INR", "JPY", "SGD", "AED"},
    "tolerance_float_eps": 1e-9,
}

REQUIRED_TXN_COLS = {"txn_id", "customer_id", "product_ids", "txn_timestamp",
                     "amount", "currency", "store_id", "payment_method"}
REQUIRED_CUST_COLS = {"customer_id", "first_name", "last_name", "email",
                      "signup_date", "country", "age"}
REQUIRED_PROD_COLS = {"product_id", "product_name", "category", "subcategory", "unit_price_usd"}

logger = logging.getLogger("customer_analytics")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"
    ))
    logger.addHandler(_h)


class PipelineError(Exception):
    """Raised when the pipeline cannot proceed due to data quality issues."""

def validate_schema(df: pd.DataFrame, required_cols: set, name: str) -> None:
    if df is None:
        raise PipelineError(f"{name} is None")
    if df.empty:
        raise PipelineError(f"{name} is empty (0 rows)")
    missing = required_cols - set(df.columns)
    if missing:
        raise PipelineError(f"{name} missing required columns: {sorted(missing)}")
    logger.info("schema_valid name=%s rows=%d cols=%d", name, len(df), len(df.columns))


def validate_currencies(transactions: pd.DataFrame) -> None:
    unknown = set(transactions["currency"].dropna().unique()) - CONFIG["valid_currencies"]
    if unknown:
        logger.warning("unknown_currencies=%s rows_affected=%d",
                       sorted(unknown),
                       int(transactions["currency"].isin(unknown).sum()))


def normalize_strings(customers: pd.DataFrame) -> pd.DataFrame:
    out = customers.copy()
    str_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        out[col] = (
            out[col]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA})
        )
    if "email" in out.columns:
        out["email"] = out["email"].str.lower()
    if "first_name" in out.columns:
        out["first_name"] = out["first_name"].str.title()
    if "last_name" in out.columns:
        out["last_name"] = out["last_name"].str.title()
    return out


def cast_and_parse(transactions: pd.DataFrame,
                   customers: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    txn = transactions.copy()
    txn["txn_timestamp"] = pd.to_datetime(txn["txn_timestamp"], errors="coerce", utc=True)

    # Strip currency symbols and thousands separators before numeric coerce
    txn["amount"] = pd.to_numeric(
        txn["amount"].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
        errors="coerce"
    )

    cust = customers.copy()
    cust["signup_date"] = pd.to_datetime(cust["signup_date"], errors="coerce", utc=True)
    cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
    cust.loc[(cust["age"] < 0) | (cust["age"] > 120), "age"] = np.nan
    now_utc = pd.Timestamp.now(tz="UTC")
    cust["signup_in_future"] = cust["signup_date"] > now_utc

    return txn, cust

def impute_nulls(customers: pd.DataFrame) -> pd.DataFrame:
    out = customers.copy()
    out["country"] = out["country"].fillna("UNKNOWN")
    global_median = out["age"].median()

    def _fill(s: pd.Series) -> pd.Series:
        m = s.median()
        return s.fillna(m if pd.notna(m) else global_median)

    out["age"] = out.groupby("country")["age"].transform(_fill)
    return out

def cap_outliers_iqr(transactions: pd.DataFrame, col: str = "amount") -> pd.DataFrame:
    out = transactions.copy()
    non_null = out[col].notna().sum()
    if non_null < 4:
        logger.warning("outlier_skip col=%s non_null=%d", col, non_null)
        out[f"{col}_capped"] = out[col]
        out[f"{col}_was_outlier"] = False
        return out

    q1, q3 = out[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    k = CONFIG["outlier_iqr_multiplier"]
    lower, upper = q1 - k * iqr, q3 + k * iqr
    out[f"{col}_capped"] = out[col].clip(lower=lower, upper=upper)
    out[f"{col}_was_outlier"] = ((out[col] < lower) | (out[col] > upper)).fillna(False)
    logger.info("outlier_cap col=%s lower=%.2f upper=%.2f flagged=%d",
                col, lower, upper, int(out[f"{col}_was_outlier"].sum()))
    return out

def convert_to_usd(transactions: pd.DataFrame, fx_rates: pd.DataFrame) -> pd.DataFrame:
    out = transactions.copy()
    out["txn_date"] = out["txn_timestamp"].dt.tz_convert("UTC").dt.normalize()

    fx = fx_rates.copy()
    fx["effective_date"] = pd.to_datetime(fx["effective_date"], utc=True).dt.normalize()
    # merge_asof requires BOTH frames sorted globally by the `on` key
    fx = fx.sort_values("effective_date").reset_index(drop=True)

    # merge_asof rejects null keys; split NaT rows out, process valid rows, recombine
    nat_mask = out["txn_date"].isna()
    nat_rows = out[nat_mask].copy()
    valid = out[~nat_mask].sort_values("txn_date").reset_index(drop=True)

    if not valid.empty:
        valid = pd.merge_asof(
            valid, fx,
            left_on="txn_date", right_on="effective_date",
            by="currency", direction="backward"
        )
    else:
        valid = valid.assign(rate_to_usd=np.nan, effective_date=pd.NaT)
    if not nat_rows.empty:
        nat_rows = nat_rows.assign(rate_to_usd=np.nan, effective_date=pd.NaT)

    out = pd.concat([valid, nat_rows], ignore_index=True)
    out["amount_usd"] = out["amount"] * out["rate_to_usd"]
    out.loc[out["currency"] == "USD", "amount_usd"] = out.loc[out["currency"] == "USD", "amount"]
    return out.drop(columns=["effective_date"], errors="ignore")


def compute_customer_tenure(customers: pd.DataFrame,
                             reference_date: pd.Timestamp) -> pd.DataFrame:
    out = customers.copy()
    ref = pd.Timestamp(reference_date)
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")
    out["days_since_signup"] = (ref - out["signup_date"]).dt.days
    out["tenure_months"] = out["days_since_signup"] / 30.4375  # avg month length
    return out


def explode_products(transactions: pd.DataFrame) -> pd.DataFrame:
    out = transactions.copy()
    out["product_ids"] = out["product_ids"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    out["basket_size"] = out["product_ids"].apply(len)
    out = out[out["basket_size"] > 0].copy()
    out["amount_per_product"] = out["amount_usd"] / out["basket_size"]
    out = out.explode("product_ids").rename(columns={"product_ids": "product_id"})
    return out.reset_index(drop=True)

def enrich_transactions(exploded: pd.DataFrame,
                        customers: pd.DataFrame,
                        products: pd.DataFrame,
                        stores: pd.DataFrame) -> pd.DataFrame:
    cust_cols = ["customer_id", "country", "age", "signup_date", "email"]
    prod_cols = ["product_id", "product_name", "category", "subcategory", "unit_price_usd"]
    store_cols = ["store_id", "region"]

    out = exploded.merge(customers[cust_cols], on="customer_id", how="left",
                         suffixes=("", "_cust"), indicator="_cust_merge")
    out["customer_missing_in_dim"] = out["_cust_merge"].eq("left_only")
    out = out.drop(columns=["_cust_merge"])

    out = out.merge(products[prod_cols], on="product_id", how="left",
                    suffixes=("", "_prod"), indicator="_prod_merge")
    out["product_missing_in_dim"] = out["_prod_merge"].eq("left_only")
    out = out.drop(columns=["_prod_merge"])

    out = out.merge(stores[store_cols], on="store_id", how="left",
                    suffixes=("", "_store"), indicator="_store_merge")
    out["store_missing_in_dim"] = out["_store_merge"].eq("left_only")
    out = out.drop(columns=["_store_merge"])
    return out

def customers_without_complaints(customers: pd.DataFrame,
                                  complaints: pd.DataFrame) -> pd.DataFrame:
    if complaints.empty:
        return customers.copy()
    complainers = set(complaints["customer_id"].dropna())
    return customers[~customers["customer_id"].isin(complainers)].copy()


def first_vs_latest_transaction(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.dropna(subset=["txn_timestamp"]).copy()
    df = df.sort_values(["customer_id", "txn_timestamp", "txn_id"])
    grouped = df.groupby("customer_id")
    first = grouped.first()[["txn_timestamp", "amount_usd"]].rename(
        columns={"txn_timestamp": "first_txn_ts", "amount_usd": "first_amount_usd"}
    )
    last = grouped.last()[["txn_timestamp", "amount_usd"]].rename(
        columns={"txn_timestamp": "latest_txn_ts", "amount_usd": "latest_amount_usd"}
    )
    out = first.join(last)
    out["gap_days"] = (out["latest_txn_ts"] - out["first_txn_ts"]).dt.days
    out["amount_delta_usd"] = out["latest_amount_usd"] - out["first_amount_usd"]
    return out.reset_index()

def compute_cumulative_spend(transactions: pd.DataFrame) -> pd.DataFrame:
    out = transactions.sort_values(["customer_id", "txn_timestamp", "txn_id"]).copy()
    out["cumulative_spend_usd"] = (
        out.groupby("customer_id")["amount_usd"]
           .apply(lambda s: s.fillna(0).cumsum())
           .reset_index(level=0, drop=True)
    )
    return out

def compute_rolling_avg(transactions: pd.DataFrame) -> pd.DataFrame:
    days = CONFIG["rolling_window_days"]
    out = transactions.sort_values(["customer_id", "txn_timestamp"]).copy()

    def _roll(group: pd.DataFrame) -> pd.Series:
        g = group.set_index("txn_timestamp")["amount_usd"]
        return g.rolling(f"{days}D", closed="both").mean().reset_index(drop=True)

    rolled = (
        out.groupby("customer_id", group_keys=False)
           .apply(lambda g: _roll(g))
    )
    out["rolling_avg_usd"] = rolled.values
    return out

def rank_products_per_customer(enriched: pd.DataFrame) -> pd.DataFrame:
    agg = (enriched.dropna(subset=["product_id"])
                   .groupby(["customer_id", "product_id"])
                   .agg(product_spend_usd=("amount_per_product", "sum"))
                   .reset_index())
    agg["product_rank"] = (
        agg.groupby("customer_id")["product_spend_usd"]
           .rank(method="dense", ascending=False)
           .astype(int)
    )
    return agg

def compute_inter_txn_gap(transactions: pd.DataFrame) -> pd.DataFrame:
    out = transactions.sort_values(["customer_id", "txn_timestamp", "txn_id"]).copy()
    out["prev_txn_ts"] = out.groupby("customer_id")["txn_timestamp"].shift(1)
    out["gap_seconds_since_prev"] = (
        (out["txn_timestamp"] - out["prev_txn_ts"]).dt.total_seconds()
    )
    return out


def assign_sessions(transactions_with_gap: pd.DataFrame) -> pd.DataFrame:
    out = transactions_with_gap.copy()
    threshold = CONFIG["session_gap_minutes"] * 60
    new_session = (out["gap_seconds_since_prev"] > threshold).fillna(False).astype(int)
    out["session_id"] = new_session.groupby(out["customer_id"]).cumsum()
    return out

def aggregate_customer_metrics(enriched: pd.DataFrame) -> pd.DataFrame:
    if enriched.empty:
        return pd.DataFrame(columns=[
            "customer_id", "txn_count", "total_spend_usd", "avg_spend_usd",
            "std_spend_usd", "distinct_product_count", "distinct_category_count",
            "first_txn_ts", "last_txn_ts", "distinct_store_count",
            "distinct_payment_method_count"
        ])

    agg = (enriched.groupby("customer_id")
                   .agg(
                        txn_count=("txn_id", "nunique"),
                        total_spend_usd=("amount_per_product", "sum"),
                        avg_spend_usd=("amount_per_product", "mean"),
                        std_spend_usd=("amount_per_product", "std"),
                        distinct_product_count=("product_id", "nunique"),
                        distinct_category_count=("category", "nunique"),
                        first_txn_ts=("txn_timestamp", "min"),
                        last_txn_ts=("txn_timestamp", "max"),
                        distinct_store_count=("store_id", "nunique"),
                        distinct_payment_method_count=("payment_method", "nunique"),
                   )
                   .reset_index())
    return agg

def pivot_monthly_spend(enriched: pd.DataFrame) -> pd.DataFrame:
    df = enriched.dropna(subset=["category"]).copy()
    df["month"] = df["txn_timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str)
    pivot = (df.pivot_table(
                 index=["customer_id", "month"],
                 columns="category",
                 values="amount_per_product",
                 aggfunc="sum",
                 fill_value=0.0,
             )
             .reset_index())
    pivot.columns.name = None
    return pivot

def bin_age_and_value(customers: pd.DataFrame,
                      customer_metrics: pd.DataFrame) -> pd.DataFrame:
    out = customers.merge(
        customer_metrics[["customer_id", "total_spend_usd"]],
        on="customer_id", how="left"
    )
    out["total_spend_usd"] = out["total_spend_usd"].fillna(0.0)

    out["age_band"] = pd.cut(out["age"],
                             bins=CONFIG["age_bins"],
                             labels=CONFIG["age_labels"],
                             right=True).astype("string")

    thresholds = CONFIG["tier_thresholds_usd"]
    bins = [thresholds["bronze"], thresholds["silver"],
            thresholds["gold"], thresholds["platinum"], np.inf]
    labels = ["bronze", "silver", "gold", "platinum"]
    out["value_tier"] = pd.cut(out["total_spend_usd"], bins=bins,
                                labels=labels, right=False, include_lowest=True).astype("string")
    return out

def category_percentiles(enriched: pd.DataFrame) -> pd.DataFrame:
    df = enriched.dropna(subset=["category"]).copy()
    out = (df.groupby("category")["amount_per_product"]
             .quantile([0.50, 0.75, 0.95])
             .unstack()
             .rename(columns={0.50: "p50", 0.75: "p75", 0.95: "p95"})
             .reset_index())
    return out

def rfm_segmentation(customer_metrics: pd.DataFrame,
                     reference_date: pd.Timestamp) -> pd.DataFrame:
    df = customer_metrics.copy()
    ref = pd.Timestamp(reference_date)
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")
    df["recency_days"] = (ref - df["last_txn_ts"]).dt.days

    def _safe_qcut(s: pd.Series, ascending_is_better: bool) -> pd.Series:
        try:
            ranks = s.rank(method="first", ascending=ascending_is_better)
            return pd.qcut(ranks, q=4, labels=[1, 2, 3, 4]).astype("Int64")
        except ValueError:
            return pd.Series([pd.NA] * len(s), index=s.index, dtype="Int64")

    df["r_score"] = _safe_qcut(df["recency_days"], ascending_is_better=False)
    df["f_score"] = _safe_qcut(df["txn_count"], ascending_is_better=True)
    df["m_score"] = _safe_qcut(df["total_spend_usd"], ascending_is_better=True)
    df["rfm_segment"] = (df["r_score"].astype("string")
                         + df["f_score"].astype("string")
                         + df["m_score"].astype("string"))
    return df[["customer_id", "recency_days", "r_score", "f_score", "m_score", "rfm_segment"]]

def cohort_retention(customers: pd.DataFrame,
                     transactions: pd.DataFrame) -> pd.DataFrame:
    cust = customers[["customer_id", "signup_date"]].dropna()
    cust = cust.copy()
    cust["cohort_month"] = cust["signup_date"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")

    txn = transactions[["customer_id", "txn_timestamp"]].dropna()
    txn = txn.copy()
    txn["activity_month"] = txn["txn_timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")

    merged = txn.merge(cust, on="customer_id", how="inner")
    merged["months_since_signup"] = (
        (merged["activity_month"] - merged["cohort_month"]).apply(lambda p: p.n)
    )
    merged = merged[merged["months_since_signup"] >= 0]

    cohort_sizes = cust.groupby("cohort_month")["customer_id"].nunique()
    active = (merged.groupby(["cohort_month", "months_since_signup"])
                    ["customer_id"].nunique()
                    .reset_index(name="active"))
    active["cohort_size"] = active["cohort_month"].map(cohort_sizes)
    active["retention"] = active["active"] / active["cohort_size"]
    active["cohort_month"] = active["cohort_month"].astype(str)
    return active

def compute_risk_score(customer_360: pd.DataFrame) -> pd.DataFrame:
    out = customer_360.copy()
    def _score(row: pd.Series) -> float:
        velocity = 30.0 if row.get("has_velocity_flag", False) else 0.0
        high_amt = 25.0 if row.get("has_high_amount_flag", False) else 0.0
        cross_border = 20.0 if row.get("has_cross_border_txn", False) else 0.0
        complaints = min(row.get("complaint_count", 0) or 0, 5) / 5.0 * 15.0
        recency_raw = row.get("recency_days", np.nan)
        if pd.isna(recency_raw):
            recency = 0.0
        else:
            # newer activity (low recency) increases risk weight
            recency = max(0.0, min(10.0, (30 - recency_raw) / 30 * 10.0))
        total = velocity + high_amt + cross_border + complaints + recency
        return float(np.clip(total, 0.0, 100.0))
    out["risk_score"] = out.apply(_score, axis=1)
    return out


def top_n_products(product_ranks: pd.DataFrame) -> pd.DataFrame:
    n = CONFIG["top_n_products_per_customer"]
    top = product_ranks[product_ranks["product_rank"] <= n].copy()
    return (top.sort_values(["customer_id", "product_rank"])
               .groupby("customer_id")
               .agg(top_products=("product_id", lambda s: list(s)),
                    top_product_spend_usd=("product_spend_usd", "sum"))
               .reset_index())

def concat_categories(enriched: pd.DataFrame) -> pd.DataFrame:
    return (enriched.dropna(subset=["category"])
                    .groupby("customer_id")["category"]
                    .agg(lambda s: ",".join(sorted(set(s))))
                    .reset_index()
                    .rename(columns={"category": "categories_concat"}))

def fraud_flags(transactions_with_gap: pd.DataFrame) -> pd.DataFrame:
    df = transactions_with_gap.copy()
    df["high_amount"] = (df["amount_usd"] >= CONFIG["fraud_amount_threshold_usd"]).fillna(False)
    df["velocity"] = (
        df["gap_seconds_since_prev"] <= CONFIG["fraud_velocity_window_seconds"]
    ).fillna(False)
    df["cross_border"] = (
        df.get("region", pd.Series([np.nan] * len(df))).notna()
        & df.get("country", pd.Series([np.nan] * len(df))).notna()
        & (df.get("region", "") != df.get("country", ""))
    )
    df["failed_payment"] = df["payment_method"].astype("string").str.upper().eq("FAILED").fillna(False)

    agg = (df.groupby("customer_id")
             .agg(has_high_amount_flag=("high_amount", "any"),
                  has_velocity_flag=("velocity", "any"),
                  has_cross_border_txn=("cross_border", "any"),
                  has_failed_payment=("failed_payment", "any"))
             .reset_index())
    return agg

def hash_customer_id(customers: pd.DataFrame) -> pd.DataFrame:
    out = customers.copy()
    def _h(row: pd.Series) -> str:
        cid = str(row.get("customer_id", "")).strip()
        email_raw = row.get("email")
        em = "<null>" if pd.isna(email_raw) else str(email_raw).strip().lower()
        return hashlib.sha256(f"{cid}|{em}".encode("utf-8")).hexdigest()
    out["customer_hash"] = out.apply(_h, axis=1)
    return out

def assign_tier(customer_metrics: pd.DataFrame) -> pd.DataFrame:
    out = customer_metrics.copy()
    spend = out["total_spend_usd"].fillna(0.0)
    txns = out["txn_count"].fillna(0)
    thr = CONFIG["tier_thresholds_usd"]

    conditions = [
        txns < CONFIG["min_transactions_for_active"],
        spend >= thr["platinum"],
        spend >= thr["gold"],
        spend >= thr["silver"],
    ]
    choices = ["inactive", "platinum", "gold", "silver"]
    out["loyalty_tier"] = np.select(conditions, choices, default="bronze")
    return out

def crosstab_payment_category(enriched: pd.DataFrame) -> pd.DataFrame:
    df = enriched.dropna(subset=["payment_method", "category"])
    ct = pd.crosstab(df["payment_method"], df["category"]).reset_index()
    ct.columns.name = None
    return ct

def run_pipeline(transactions: pd.DataFrame,
                 customers: pd.DataFrame,
                 products: pd.DataFrame,
                 stores: pd.DataFrame,
                 fx_rates: pd.DataFrame,
                 complaints: pd.DataFrame,
                 historical_transactions: Optional[pd.DataFrame] = None,
                 reference_date: Optional[pd.Timestamp] = None) -> Dict[str, pd.DataFrame]:

    if reference_date is None:
        reference_date = pd.Timestamp.now(tz="UTC")
    logger.info("pipeline_start reference_date=%s", reference_date.isoformat())

    validate_schema(transactions, REQUIRED_TXN_COLS, "transactions")
    validate_schema(customers, REQUIRED_CUST_COLS, "customers")
    validate_schema(products, REQUIRED_PROD_COLS, "products")
    validate_currencies(transactions)

    customers = normalize_strings(customers)                                    # T01
    transactions, customers = cast_and_parse(transactions, customers)            # T02
    customers = impute_nulls(customers)                                          # T03
    transactions = cap_outliers_iqr(transactions, col="amount")                  # T04
    transactions = convert_to_usd(transactions, fx_rates)                        # T05
    customers = compute_customer_tenure(customers, reference_date)               # T06

    if historical_transactions is not None and not historical_transactions.empty:
        transactions = union_with_historical(transactions, historical_transactions)  # T30
        logger.info("post_union_rows=%d", len(transactions))

    transactions = deduplicate_latest(transactions, key="txn_id",
                                       order_by="txn_timestamp")                 # T24
    exploded = explode_products(transactions)                                    # T07
    enriched = enrich_transactions(exploded, customers, products, stores)        # T08

    transactions_ordered = compute_cumulative_spend(transactions)                # T11
    transactions_ordered = compute_rolling_avg(transactions_ordered)             # T12
    transactions_ordered = compute_inter_txn_gap(transactions_ordered)           # T14

    transactions_ordered = transactions_ordered.merge(
        stores[["store_id", "region"]], on="store_id", how="left"
    ).merge(
        customers[["customer_id", "country"]], on="customer_id", how="left"
    )
    transactions_ordered = assign_sessions(transactions_ordered)                 # T15

    customer_metrics = aggregate_customer_metrics(enriched)                       # T16
    monthly_pivot = pivot_monthly_spend(enriched)                                 # T17
    customers_binned = bin_age_and_value(customers, customer_metrics)             # T18
    cat_pct = category_percentiles(enriched)                                      # T19
    rfm = rfm_segmentation(customer_metrics, reference_date)                      # T20
    cohorts = cohort_retention(customers, transactions)                           # T21
    product_ranks = rank_products_per_customer(enriched)                          # T13
    top_n = top_n_products(product_ranks)                                         # T23
    categories_concat = concat_categories(enriched)                               # T25
    flags = fraud_flags(transactions_ordered)                                     # T26
    no_complaint_customers = customers_without_complaints(customers, complaints)  # T09
    first_latest = first_vs_latest_transaction(transactions_ordered)              # T10
    crosstab = crosstab_payment_category(enriched)                                # T29
    hashed = hash_customer_id(customers)                                          # T27

    complaint_counts = (complaints.groupby("customer_id")
                                  .size()
                                  .reset_index(name="complaint_count")
                        if not complaints.empty
                        else pd.DataFrame(columns=["customer_id", "complaint_count"]))

    customer_360 = (customers_binned
                    .merge(customer_metrics, on="customer_id", how="left",
                           suffixes=("", "_dup"))
                    .merge(rfm, on="customer_id", how="left")
                    .merge(flags, on="customer_id", how="left")
                    .merge(top_n, on="customer_id", how="left")
                    .merge(categories_concat, on="customer_id", how="left")
                    .merge(complaint_counts, on="customer_id", how="left")
                    .merge(hashed[["customer_id", "customer_hash"]],
                           on="customer_id", how="left"))
    customer_360 = customer_360.loc[:, ~customer_360.columns.str.endswith("_dup")]

    customer_360["complaint_count"] = customer_360["complaint_count"].fillna(0).astype(int)
    for c in ["has_high_amount_flag", "has_velocity_flag",
              "has_cross_border_txn", "has_failed_payment"]:
        if c in customer_360.columns:
            customer_360[c] = customer_360[c].fillna(False)

    customer_360 = assign_tier(customer_360)                                      # T28
    customer_360 = compute_risk_score(customer_360)                               # T22

    logger.info("pipeline_complete customer_360_rows=%d", len(customer_360))

    return {
        "customer_360": customer_360,
        "transactions_features": transactions_ordered,
        "enriched_lines": enriched,
        "monthly_pivot": monthly_pivot,
        "category_percentiles": cat_pct,
        "cohort_retention": cohorts,
        "product_ranks": product_ranks,
        "first_vs_latest": first_latest,
        "crosstab_payment_category": crosstab,
        "customers_without_complaints": no_complaint_customers,
    }

if __name__ == "__main__":
    from code_example.generate_synthetic_data import generate_synthetic_data

    data = generate_synthetic_data(seed=42)
    outputs = run_pipeline(
        transactions=data["transactions"],
        customers=data["customers"],
        products=data["products"],
        stores=data["stores"],
        fx_rates=data["fx_rates"],
        complaints=data["complaints"],
        reference_date=pd.Timestamp("2025-01-01", tz="UTC"),
    )
    for name, frame in outputs.items():
        logger.info("output name=%s rows=%d cols=%d",
                    name, len(frame), len(frame.columns))
        print(f"\n=== {name} (head) ===")
        print(frame.head(3).to_string())
