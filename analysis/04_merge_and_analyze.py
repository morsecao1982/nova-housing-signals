"""
Merge restaurant signals with housing data.
Test lag relationships and build predictive model.

Output: data/merged_analysis.csv, data/lag_correlations.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DATA = Path(__file__).parent.parent / "data"

# ── 1. Load both signals ──────────────────────────────────────────────────
print("Loading signals...")
rest = pd.read_csv(DATA / "restaurant_signals.csv", parse_dates=["month"])
hous = pd.read_csv(DATA / "housing_signals.csv",    parse_dates=["month"])

print(f"Restaurant signals: {rest.shape}")
print(f"Housing signals:    {hous.shape}")

# ── 2. Merge on city + month ──────────────────────────────────────────────
merged = pd.merge(rest, hous, on=["city", "month"], how="inner")
merged = merged.dropna(subset=["zhvi", "review_volume"])
# Normalize both to first-of-month before merge
rest["month"] = pd.to_datetime(rest["month"]).dt.to_period("M").dt.to_timestamp()
hous["month"] = pd.to_datetime(hous["month"]).dt.to_period("M").dt.to_timestamp()

merged = pd.merge(rest, hous, on=["city", "month"], how="inner")
merged = merged.dropna(subset=["zhvi", "review_volume"])
print(f"Merged: {merged.shape}")
print(f"Date range: {merged['month'].min().date()} → {merged['month'].max().date()}\n")

# ── 3. Lag correlation analysis ───────────────────────────────────────────
# Test: do restaurant signals at time t predict housing at t+lag?
print("── Lag Correlation Analysis ──")
print("(Positive correlation = restaurant signal leads housing price change)\n")

RESTAURANT_SIGNALS = [
    "review_volume",
    "review_yoy_growth",
    "checkin_volume",
    "checkin_yoy_growth",
    "avg_stars",
    "reviews_per_restaurant",
]
HOUSING_TARGETS = ["zhvi_3m_fwd", "zhvi_6m_fwd", "zhvi_12m_fwd"]
LAGS = [0, 1, 2, 3, 6, 9, 12]

results = []
for city in merged["city"].unique():
    df = merged[merged["city"] == city].sort_values("month")
    for sig in RESTAURANT_SIGNALS:
        for target in HOUSING_TARGETS:
            for lag in LAGS:
                x = df[sig].shift(lag)
                y = df[target]
                valid = x.notna() & y.notna()
                if valid.sum() < 24:
                    continue
                r, p = stats.pearsonr(x[valid], y[valid])
                results.append({
                    "city": city,
                    "restaurant_signal": sig,
                    "housing_target": target,
                    "lag_months": lag,
                    "correlation": round(r, 3),
                    "p_value": round(p, 4),
                    "significant": p < 0.05,
                    "n": int(valid.sum()),
                })

lag_df = pd.DataFrame(results)

if lag_df.empty:
    print("No results — check merge step.")
else:
    sig = lag_df[lag_df["significant"]]
    print("Top 20 strongest significant correlations:")
    top = sig.reindex(sig["correlation"].abs().sort_values(ascending=False).index).head(20)
    print(top[["city","restaurant_signal","housing_target","lag_months","correlation","p_value"]].to_string(index=False))

# ── 4. Best lag per city ──────────────────────────────────────────────────
print("\n── Best predictive signal per city (6-month housing target) ──")
if not lag_df.empty:
    sub = lag_df[(lag_df["housing_target"] == "zhvi_6m_fwd") & lag_df["significant"]]
    if not sub.empty:
        best = (
            sub.reindex(sub["correlation"].abs().sort_values(ascending=False).index)
            .groupby("city").first().reset_index()
        )
        print(best[["city","restaurant_signal","lag_months","correlation"]].to_string(index=False))

# ── 5. Simple predictive model ────────────────────────────────────────────
print("\n── Predictive Model: Review YoY Growth → 6-Month Housing Change ──")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
# sklearn might not be installed, fallback to statsmodels
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    model_results = []
    for city in merged["city"].unique():
        df = merged[merged["city"] == city].sort_values("month").copy()
        df = df.dropna(subset=["review_yoy_growth", "checkin_yoy_growth", "zhvi_6m_fwd"])

        # Train on pre-2020, test on 2020+
        train = df[df["month"] < "2020-01-01"]
        test  = df[df["month"] >= "2020-01-01"]

        if len(train) < 20 or len(test) < 6:
            continue

        features = ["review_yoy_growth", "checkin_yoy_growth"]
        # Replace inf values and drop NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        train = df[df["month"] < "2020-01-01"].dropna(subset=features + ["zhvi_6m_fwd"])
        test  = df[df["month"] >= "2020-01-01"].dropna(subset=features + ["zhvi_6m_fwd"])
        if len(train) < 20 or len(test) < 6:
            continue
        X_train, y_train = train[features], train["zhvi_6m_fwd"]
        X_test,  y_test  = test[features],  test["zhvi_6m_fwd"]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        mae  = np.mean(np.abs(y_test - y_pred))

        model_results.append({
            "city": city,
            "train_samples": len(train),
            "test_samples":  len(test),
            "r2_test":  round(r2, 3),
            "mae_test": round(mae, 2),
            "coef_review_yoy":  round(model.coef_[0], 4),
            "coef_checkin_yoy": round(model.coef_[1], 4),
        })

    model_df = pd.DataFrame(model_results)
    print(model_df.to_string(index=False))

except ImportError:
    print("sklearn not installed — skipping model. Run: pip install scikit-learn")

# ── 6. Save outputs ───────────────────────────────────────────────────────
merged.to_csv(DATA / "merged_analysis.csv", index=False)
lag_df.to_csv(DATA / "lag_correlations.csv", index=False)
print(f"\nSaved → data/merged_analysis.csv")
print(f"Saved → data/lag_correlations.csv")
