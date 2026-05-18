"""
Process Zillow ZHVI data for our 5 target metros.
Computes MoM, YoY, and forward-looking changes (prediction targets).

Output: data/housing_signals.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

METROS = {
    "Philadelphia, PA": "Philadelphia",
    "Tampa, FL":        "Tampa",
    "Indianapolis, IN": "Indianapolis",
    "Nashville, TN":    "Nashville",
    "New Orleans, LA":  "New Orleans",
}

# ── 1. Load metro ZHVI ────────────────────────────────────────────────────
print("Loading Zillow metro data...")
raw = pd.read_csv(DATA / "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")

# Filter to our metros
metro_df = raw[raw["RegionName"].isin(METROS.keys())].copy()
metro_df["city"] = metro_df["RegionName"].map(METROS)

# Melt wide → long
date_cols = [c for c in metro_df.columns if c.startswith("20")]
housing = metro_df[["city", "RegionName"] + date_cols].melt(
    id_vars=["city", "RegionName"],
    var_name="month",
    value_name="zhvi",
)
housing["month"] = pd.to_datetime(housing["month"]).dt.to_period("M").dt.to_timestamp()
housing = housing.sort_values(["city", "month"]).reset_index(drop=True)

# Filter to analysis period
housing = housing[(housing["month"] >= "2004-01-01") & (housing["month"] <= "2026-01-01")]

print(f"Housing data shape: {housing.shape}")
print(f"Date range: {housing['month'].min().date()} → {housing['month'].max().date()}")

# ── 2. Compute price change features ─────────────────────────────────────
for city in METROS.values():
    mask = housing["city"] == city

    # Month-over-month % change
    housing.loc[mask, "zhvi_mom"] = (
        housing.loc[mask, "zhvi"].pct_change(1).mul(100)
    )
    # Year-over-year % change
    housing.loc[mask, "zhvi_yoy"] = (
        housing.loc[mask, "zhvi"].pct_change(12).mul(100)
    )
    # 3-month forward change (short-term prediction target)
    housing.loc[mask, "zhvi_3m_fwd"] = (
        housing.loc[mask, "zhvi"].pct_change(3).shift(-3).mul(100)
    )
    # 6-month forward change (medium-term target)
    housing.loc[mask, "zhvi_6m_fwd"] = (
        housing.loc[mask, "zhvi"].pct_change(6).shift(-6).mul(100)
    )
    # 12-month forward change (annual target)
    housing.loc[mask, "zhvi_12m_fwd"] = (
        housing.loc[mask, "zhvi"].pct_change(12).shift(-12).mul(100)
    )
    # 3-month rolling avg to smooth noise
    housing.loc[mask, "zhvi_3m_avg"] = (
        housing.loc[mask, "zhvi"].rolling(3).mean()
    )

# ── 3. Summary ────────────────────────────────────────────────────────────
print("\n── Housing price summary by city ──")
summary = housing.groupby("city").agg(
    min_zhvi=("zhvi", "min"),
    max_zhvi=("zhvi", "max"),
    avg_yoy=("zhvi_yoy", "mean"),
    latest_zhvi=("zhvi", "last"),
).round(1)
print(summary.to_string())

# ── 4. Save ───────────────────────────────────────────────────────────────
out_path = DATA / "housing_signals.csv"
housing.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
