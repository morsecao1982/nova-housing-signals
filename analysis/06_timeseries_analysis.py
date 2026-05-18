"""
Proper time-series analysis:
1. Stationarity tests (ADF)
2. Differencing to achieve stationarity
3. Granger causality tests
4. VAR model with walk-forward validation
5. Honest out-of-sample evaluation (2016-2019, pre-COVID)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

DATA = Path(__file__).parent.parent / "data"
df   = pd.read_csv(DATA / "merged_analysis.csv", parse_dates=["month"])
df   = df.replace([np.inf, -np.inf], np.nan)
df   = df.sort_values(["city", "month"])

CITIES = sorted(df["city"].unique())

# ── Helper: ADF test ───────────────────────────────────────────────────────
def adf_test(series, name=""):
    s = series.dropna()
    if len(s) < 20:
        return None, None
    result = adfuller(s, autolag="AIC")
    pval   = result[1]
    stat   = "STATIONARY ✓" if pval < 0.05 else "NON-STATIONARY ✗"
    return pval, stat

# ── 1. Stationarity Tests ─────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: STATIONARITY TESTS (ADF)")
print("Null hypothesis: series has a unit root (non-stationary)")
print("p < 0.05 → reject null → STATIONARY")
print("=" * 70)

SIGNALS = ["review_volume", "reviews_per_restaurant",
           "checkin_volume", "avg_stars", "zhvi", "zhvi_mom", "zhvi_yoy"]

stationarity_results = []
for city in CITIES:
    cdf = df[df["city"] == city].set_index("month")
    print(f"\n  {city}:")
    for sig in SIGNALS:
        if sig not in cdf.columns:
            continue
        pval, stat = adf_test(cdf[sig], sig)
        if pval is None:
            continue
        print(f"    {sig:30s}  p={pval:.4f}  {stat}")
        stationarity_results.append({"city": city, "series": sig,
                                     "p_value": pval, "stationary": pval < 0.05})

stat_df = pd.DataFrame(stationarity_results)
print(f"\n  Summary — % stationary per series:")
print(stat_df.groupby("series")["stationary"].mean().mul(100).round(0).to_string())

# ── 2. Build Stationary Series ────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("STEP 2: TRANSFORMATIONS TO ACHIEVE STATIONARITY")
print("=" * 70)

# Housing: use MoM % change (already computed as zhvi_mom)
# Restaurant: use MoM % change of review_volume and reviews_per_restaurant

for city in CITIES:
    mask = df["city"] == city
    # MoM % change of restaurant signals
    df.loc[mask, "review_mom"]     = df.loc[mask, "review_volume"].pct_change(1).mul(100)
    df.loc[mask, "rpr_mom"]        = df.loc[mask, "reviews_per_restaurant"].pct_change(1).mul(100)
    df.loc[mask, "checkin_mom"]    = df.loc[mask, "checkin_volume"].pct_change(1).mul(100)
    # YoY change (removes seasonality)
    df.loc[mask, "review_yoy"]     = df.loc[mask, "review_volume"].pct_change(12).mul(100)
    df.loc[mask, "rpr_yoy"]        = df.loc[mask, "reviews_per_restaurant"].pct_change(12).mul(100)

STATIONARY_SIGNALS = ["review_mom", "rpr_mom", "review_yoy", "rpr_yoy"]

print("\n  Re-testing transformed series:")
for city in CITIES:
    cdf = df[df["city"] == city].set_index("month")
    print(f"\n  {city}:")
    for sig in STATIONARY_SIGNALS + ["zhvi_mom", "zhvi_yoy"]:
        if sig not in cdf.columns:
            continue
        pval, stat = adf_test(cdf[sig])
        if pval is not None:
            print(f"    {sig:20s}  p={pval:.4f}  {stat}")

# ── 3. Granger Causality Tests ────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("STEP 3: GRANGER CAUSALITY TESTS")
print("Null: restaurant signal does NOT Granger-cause housing change")
print("p < 0.05 → reject null → restaurant DOES lead housing ✓")
print("=" * 70)

granger_results = []
MAX_LAG = 12

for city in CITIES:
    cdf = df[df["city"] == city].set_index("month").sort_index()
    print(f"\n  {city}:")

    for sig in ["review_mom", "rpr_mom", "review_yoy", "rpr_yoy"]:
        for housing_target in ["zhvi_mom", "zhvi_yoy"]:
            pair = cdf[[housing_target, sig]].dropna()
            if len(pair) < MAX_LAG + 10:
                continue
            try:
                gc = grangercausalitytests(pair, maxlag=MAX_LAG, verbose=False)
                # Find best (min p-value) lag
                best_lag  = min(gc, key=lambda l: gc[l][0]["ssr_ftest"][1])
                best_pval = gc[best_lag][0]["ssr_ftest"][1]
                sig_flag  = "✓ SIGNIFICANT" if best_pval < 0.05 else "✗"
                print(f"    {sig:15s} → {housing_target:12s}  "
                      f"best lag={best_lag:2d}m  p={best_pval:.4f}  {sig_flag}")
                granger_results.append({
                    "city": city, "signal": sig,
                    "housing": housing_target,
                    "best_lag": best_lag,
                    "p_value": best_pval,
                    "significant": best_pval < 0.05,
                })
            except (InfeasibleTestError, Exception):
                pass

gc_df = pd.DataFrame(granger_results)
print(f"\n  Summary — % of city/signal pairs significant:")
if not gc_df.empty:
    print(gc_df.groupby("signal")["significant"].mean().mul(100).round(0).to_string())

# ── 4. VAR Model with Walk-Forward Validation ─────────────────────────────
print("\n\n" + "=" * 70)
print("STEP 4: VAR MODEL — WALK-FORWARD VALIDATION (2016–2019)")
print("Train on rolling window, predict 1/3/6 months ahead")
print("Evaluation period: 2016-01 to 2019-12 (normal market, pre-COVID)")
print("=" * 70)

var_results = []

for city in CITIES:
    cdf = df[df["city"] == city].set_index("month").sort_index()

    # Use YoY changes (more stable than MoM)
    var_data = cdf[["zhvi_yoy", "rpr_yoy", "review_yoy"]].dropna()

    if len(var_data) < 48:
        print(f"\n  {city}: insufficient data, skipping")
        continue

    print(f"\n  {city}:")
    eval_start = pd.Timestamp("2016-01-01")
    eval_end   = pd.Timestamp("2019-12-31")
    eval_idx   = var_data.index[(var_data.index >= eval_start) &
                                  (var_data.index <= eval_end)]

    preds_1m, preds_6m, actuals_1m, actuals_6m = [], [], [], []

    for t in eval_idx:
        # Use all data up to but not including t
        train_data = var_data[var_data.index < t]
        if len(train_data) < 24:
            continue

        try:
            # Select lag order (max 6 to avoid overfitting)
            model = VAR(train_data)
            lag_order = model.select_order(maxlags=6)
            best_lag  = lag_order.aic
            best_lag  = max(1, min(best_lag, 6))

            fitted = model.fit(best_lag)

            # Forecast
            fc = fitted.forecast(train_data.values[-best_lag:], steps=6)
            fc_df = pd.DataFrame(fc, columns=var_data.columns)

            # 1-month ahead
            pred_1m = fc_df["zhvi_yoy"].iloc[0]
            # 6-month ahead (cumulative direction)
            pred_6m = fc_df["zhvi_yoy"].iloc[5]

            # Actuals
            future = var_data[var_data.index > t]
            if len(future) >= 1:
                act_1m = future["zhvi_yoy"].iloc[0]
                preds_1m.append(pred_1m)
                actuals_1m.append(act_1m)
            if len(future) >= 6:
                act_6m = future["zhvi_yoy"].iloc[5]
                preds_6m.append(pred_6m)
                actuals_6m.append(act_6m)
        except Exception:
            continue

    for horizon, preds, actuals in [
        ("1-month",  preds_1m,  actuals_1m),
        ("6-month",  preds_6m,  actuals_6m),
    ]:
        if len(preds) < 6:
            continue
        p = np.array(preds)
        a = np.array(actuals)
        r2   = r2_score(a, p) if len(set(a)) > 1 else np.nan
        mae  = mean_absolute_error(a, p)
        rmse = np.sqrt(np.mean((a - p) ** 2))
        dacc = np.mean(np.sign(a) == np.sign(p)) * 100
        corr = np.corrcoef(p, a)[0, 1] if len(p) > 2 else np.nan

        print(f"    {horizon:8s}  R²={r2:6.3f}  MAE={mae:.2f}  "
              f"RMSE={rmse:.2f}  Dir.Acc={dacc:.0f}%  Corr={corr:.3f}  n={len(p)}")
        var_results.append({
            "city": city, "horizon": horizon,
            "R²": round(r2, 3), "MAE": round(mae, 2),
            "RMSE": round(rmse, 2), "Dir.Acc%": round(dacc, 1),
            "Corr": round(corr, 3), "n": len(p),
        })

# ── 5. Summary ────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("STEP 5: SUMMARY")
print("=" * 70)

if var_results:
    vr_df = pd.DataFrame(var_results)
    print("\n  VAR model — average across cities:")
    print(vr_df.groupby("horizon")[["R²","MAE","Dir.Acc%","Corr"]].mean().round(3).to_string())

if not gc_df.empty:
    sig_gc = gc_df[gc_df["significant"]]
    print(f"\n  Granger causality: {len(sig_gc)}/{len(gc_df)} pairs significant "
          f"({len(sig_gc)/len(gc_df)*100:.0f}%)")
    if not sig_gc.empty:
        print(f"  Best signal overall: "
              f"{sig_gc.loc[sig_gc['p_value'].idxmin(), 'signal']} → "
              f"{sig_gc.loc[sig_gc['p_value'].idxmin(), 'housing']} "
              f"(lag {sig_gc.loc[sig_gc['p_value'].idxmin(), 'best_lag']}m, "
              f"p={sig_gc['p_value'].min():.4f})")

print(f"""
  Key:
  R² > 0.3   = Moderate predictive power
  R² > 0.5   = Strong predictive power
  Dir.Acc%   = % correct on direction (up/down). Baseline = 50%
  Corr       = Pearson correlation between predicted and actual
  Granger p  = Probability restaurant does NOT lead housing
""")

# Save
if var_results:
    pd.DataFrame(var_results).to_csv(DATA / "var_performance.csv", index=False)
if granger_results:
    gc_df.to_csv(DATA / "granger_results.csv", index=False)
print("Saved → data/var_performance.csv, data/granger_results.csv")
