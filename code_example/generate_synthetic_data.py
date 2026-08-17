from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd


def generate_synthetic_data(seed: int = 42,
                             n_customers: int = 50,
                             n_txn: int = 500,
                             n_products: int = 30,
                             n_stores: int = 20,
                             n_complaints: int = 15) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    customers = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n_customers)],
        "first_name": rng.choice(["alice", "  bob ", "Charlie", "DIANA",
                                   "  ", None, "Élise"], n_customers),
        "last_name": rng.choice(["smith", "JONES", "  o'connor ", None, ""],
                                n_customers),
        "email": rng.choice(["A@X.com", "b@y.com ", " C@Z.COM",
                              None, "  ", "invalid"], n_customers),
        "signup_date": pd.to_datetime(
            rng.choice(pd.date_range("2022-01-01", "2024-12-01", freq="D"),
                       n_customers)
        ),
        "country": rng.choice(["US", "UK", "IN", "SG", None, "  "], n_customers),
        "age": rng.choice([25, 35, 45, 60, np.nan, -5, 150, 18], n_customers),
    })

    product_ids = [f"P{i:03d}" for i in range(n_products)]
    products = pd.DataFrame({
        "product_id": product_ids,
        "product_name": [f"Product {i}" for i in range(n_products)],
        "category": rng.choice(["fuel", "convenience", "lubricants",
                                 "carwash", "food", None], n_products),
        "subcategory": rng.choice(["premium", "regular", "diesel", None],
                                   n_products),
        "unit_price_usd": rng.uniform(1, 200, n_products).round(2),
    })

    stores = pd.DataFrame({
        "store_id": [f"S{i:03d}" for i in range(n_stores)],
        "region": rng.choice(["US", "UK", "IN", "SG", "AE"], n_stores),
        "opening_date": pd.to_datetime("2020-01-01"),
    })

    fx_rates = pd.DataFrame({
        "currency": ["EUR", "GBP", "INR", "JPY", "SGD", "AED", "USD"] * 3,
        "rate_to_usd": [1.08, 1.27, 0.012, 0.0067, 0.74, 0.27, 1.0,
                        1.09, 1.28, 0.012, 0.0068, 0.74, 0.27, 1.0,
                        1.10, 1.29, 0.013, 0.0068, 0.75, 0.27, 1.0],
        "effective_date": (["2023-01-01"] * 7
                           + ["2023-06-01"] * 7
                           + ["2024-01-01"] * 7),
    })

    txn_dates = pd.to_datetime(
        rng.choice(pd.date_range("2023-01-01", "2024-12-01", freq="h"), n_txn)
    ).tz_localize("UTC")
    transactions = pd.DataFrame({
        "txn_id": [f"T{i:06d}" for i in range(n_txn)],
        "customer_id": rng.choice(customers["customer_id"], n_txn),
        "product_ids": [list(rng.choice(product_ids,
                                         size=rng.integers(1, 5),
                                         replace=False))
                        for _ in range(n_txn)],
        "txn_timestamp": txn_dates,
        "amount": rng.choice(
            [10.5, 25.0, 100.0, 250.0, 50000.0,
             "$1,234.50", -15.0, np.nan, "bad"],
            n_txn
        ),
        "currency": rng.choice(["USD", "EUR", "GBP", "INR", "JPY", "XYZ"], n_txn),
        "store_id": rng.choice(stores["store_id"], n_txn),
        "payment_method": rng.choice(["CARD", "CASH", "WALLET", "FAILED", None],
                                      n_txn),
    })

    complaints = pd.DataFrame({
        "complaint_id": [f"CMP{i:04d}" for i in range(n_complaints)],
        "customer_id": rng.choice(customers["customer_id"], n_complaints),
        "complaint_date": pd.to_datetime(rng.choice(
            pd.date_range("2023-01-01", "2024-12-01", freq="D"),
            n_complaints)),
        "severity": rng.choice(["low", "medium", "high"], n_complaints),
    })

    return {
        "transactions": transactions,
        "customers": customers,
        "products": products,
        "stores": stores,
        "fx_rates": fx_rates,
        "complaints": complaints,
    }


if __name__ == "__main__":
    data = generate_synthetic_data(seed=42)
    print(f"Generated {len(data)} frames:")
    for name, frame in data.items():
        print(f"\n--- {name} (rows={len(frame)}, cols={len(frame.columns)}) ---")
        print(f"  dtypes: {dict(frame.dtypes.astype(str))}")
        print(frame.head(3).to_string())
