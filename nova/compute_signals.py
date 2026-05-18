"""
Compute monthly restaurant signals for NoVA from snapshots.
Handles bootstrap period (< 12 months) and mature period (12+ months).

Output: data/nova_signals.csv
"""
import json, os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA     = Path(__file__).parent.parent / "data"
SNAP_DIR = DATA / "nova_snapshots"

def load_snapshots():
    rows = []
    for f in sorted(SNAP_DIR.glob("*.json")):
        with open(f) as fp:
            snap = json.load(fp)
        month = snap["month"]
        for area, restaurants in snap["areas"].items():
            if not restaurants: continue
            n    = len(restaurants)
            ratings   = [r["rating"] for r in restaurants if r.get("rating")]
            reviews   = [r["review_count"] for r in restaurants if r.get("review_count")]
            prices    = [r["price_level"] for r in restaurants
                         if r.get("price_level") is not None]
            rows.append({
                "month":          month,
                "area":           area,
                "n_restaurants":  n,
                "total_reviews":  sum(reviews),
                "avg_stars":      np.mean(ratings) if ratings else np.nan,
                "avg_price_tier": np.mean(prices) if prices else np.nan,
                "pct_budget":     sum(1 for p in prices if p <= 1) / len(prices) if prices else np.nan,
                "pct_upscale":    sum(1 for p in prices if p >= 3) / len(prices) if prices else np.nan,
                "reviews_per_restaurant": sum(reviews) / n if n > 0 else 0,
            })
    return pd.DataFrame(rows)

def compute_signals(df):
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["area","month"]).reset_index(drop=True)

    n_months = df["month"].nunique()
    print(f"  Snapshots available: {n_months} month(s)")

    for area in df["area"].unique():
        m = df["area"] == area
        # YoY (needs 12 months)
        df.loc[m, "review_yoy"]   = df.loc[m, "total_reviews"].pct_change(12).mul(100)
        df.loc[m, "rpr_yoy"]      = df.loc[m, "reviews_per_restaurant"].pct_change(12).mul(100)
        df.loc[m, "checkin_yoy"]  = df.loc[m, "total_reviews"].pct_change(12).mul(100)  # proxy
        # MoM (available after 1 month)
        df.loc[m, "review_mom"]   = df.loc[m, "total_reviews"].pct_change(1).mul(100)

        # Bootstrap: fill YoY with scaled MoM when not enough history
        if n_months < 12:
            df.loc[m, "review_yoy"]  = df.loc[m, "review_mom"] * 12   # annualize
            df.loc[m, "rpr_yoy"]     = df.loc[m, "review_mom"] * 12
            df.loc[m, "checkin_yoy"] = df.loc[m, "review_mom"] * 12

    # Lag features
    for area in df["area"].unique():
        m = df["area"] == area
        for sig in ["review_yoy","rpr_yoy","checkin_yoy","review_mom",
                    "avg_stars","avg_price_tier","pct_budget","pct_upscale"]:
            df.loc[m, f"{sig}_t0"] = df.loc[m, sig]
            df.loc[m, f"{sig}_t1"] = df.loc[m, sig].shift(1)
            df.loc[m, f"{sig}_t3"] = df.loc[m, sig].shift(3)

    return df

def run():
    print("Loading snapshots...")
    df = load_snapshots()
    if df.empty:
        print("No snapshots found. Run collect_data.py first.")
        return None

    print(f"  Areas: {sorted(df['area'].unique())}")
    df = compute_signals(df)
    out = DATA / "nova_signals.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows → {out}")
    return df

if __name__ == "__main__":
    run()
