"""
Direct multi-step forecasting with cross-city validation.

Key improvements over previous model:
1. Full 2005-2022 period — includes 2008-2014 housing downturn
2. Cross-city validation — train on N cities, test on held-out city
3. Direct forecasting — one model per horizon (no error compounding)
4. Proper features — lagged restaurant signals at multiple lookback windows
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DATA = Path(__file__).parent.parent / "data"
df   = pd.read_csv(DATA / "merged_analysis.csv", parse_dates=["month"])
df   = df.replace([np.inf, -np.inf], np.nan).sort_values(["city", "month"])

CITIES   = sorted(df["city"].unique())
HORIZONS = [1, 3, 6, 12]   # months ahead

# ── 1. Compute stationary restaurant signals ──────────────────────────────
print("Building features...")
for city in CITIES:
    m = df["city"] == city
    df.loc[m, "review_yoy"] = df.loc[m, "review_volume"].pct_change(12).mul(100)
    df.loc[m, "rpr_yoy"]    = df.loc[m, "reviews_per_restaurant"].pct_change(12).mul(100)
    df.loc[m, "checkin_yoy"]= df.loc[m, "checkin_volume"].pct_change(12).mul(100)
    df.loc[m, "review_mom"] = df.loc[m, "review_volume"].pct_change(1).mul(100)

# ── 2. Build feature matrix ───────────────────────────────────────────────
# For each city/month, create lag features (restaurant signals at t-0, t-1, t-3)
# and forward housing targets (zhvi change at t+H)

def build_features(city_df, horizon):
    """
    For each row t: features = restaurant signals at t, t-1, t-3
                    target   = housing % change from t to t+horizon
    """
    d = city_df.sort_values("month").copy()

    # Forward target: % change in ZHVI over next H months
    d[f"target_{horizon}m"] = d["zhvi"].pct_change(horizon).shift(-horizon).mul(100)

    # Feature: restaurant signals at t
    feat_cols = []
    for sig in ["review_yoy", "rpr_yoy", "checkin_yoy", "review_mom", "avg_stars"]:
        if sig not in d.columns:
            continue
        d[f"{sig}_t0"] = d[sig]
        d[f"{sig}_t1"] = d[sig].shift(1)
        d[f"{sig}_t3"] = d[sig].shift(3)
        feat_cols += [f"{sig}_t0", f"{sig}_t1", f"{sig}_t3"]

    d = d.dropna(subset=feat_cols + [f"target_{horizon}m"])
    return d, feat_cols, f"target_{horizon}m"

# ── 3. Check period coverage ──────────────────────────────────────────────
print("\n── Data coverage by city and period ──")
for city in CITIES:
    sub = df[df["city"] == city].dropna(subset=["review_yoy","zhvi"])
    yr  = sub["month"].dt.year
    print(f"  {city:15s}: {sub['month'].min().date()} → {sub['month'].max().date()} "
          f"  ({len(sub)} months)  "
          f"  2008-2014: {((yr>=2008)&(yr<=2014)).sum()} months")

# ── 4. Cross-city leave-one-out validation ────────────────────────────────
print("\n\n" + "=" * 70)
print("CROSS-CITY LEAVE-ONE-OUT VALIDATION")
print("Train on 4 cities → Test on held-out city")
print("Includes full 2005-2022 (housing crash + recovery + COVID)")
print("=" * 70)

all_results = []

for horizon in HORIZONS:
    print(f"\n── Horizon: {horizon}-month ahead ──")

    city_datasets = {}
    for city in CITIES:
        cdf = df[df["city"] == city].copy()
        d, feat_cols, target_col = build_features(cdf, horizon)
        city_datasets[city] = (d, feat_cols, target_col)

    for test_city in CITIES:
        # Train on all other cities
        train_dfs = [city_datasets[c][0] for c in CITIES if c != test_city]
        train_all = pd.concat(train_dfs, ignore_index=True)

        test_d, feat_cols, target_col = city_datasets[test_city]

        if len(train_all) < 50 or len(test_d) < 20:
            continue

        # Clean inf/nan
        train_clean = train_all.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols + [target_col])
        test_clean  = test_d.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols + [target_col])

        if len(train_clean) < 50 or len(test_clean) < 20:
            continue

        X_train = train_clean[feat_cols].values
        y_train = train_clean[target_col].values
        X_test  = test_clean[feat_cols].values
        y_test  = test_clean[target_col].values
        months  = test_clean["month"].values

        # Scale
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train)
        X_test_s  = sc.transform(X_test)

        models = {
            "Ridge":    Ridge(alpha=1.0),
            "RandForest": RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42),
            "GradBoost": GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                    learning_rate=0.05, random_state=42),
        }

        for model_name, model in models.items():
            if model_name == "Ridge":
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            r2   = r2_score(y_test, y_pred)
            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            dacc = np.mean(np.sign(y_test) == np.sign(y_pred)) * 100
            corr = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 2 else np.nan

            # Period breakdown
            test_years = pd.to_datetime(months).year
            crisis    = (test_years >= 2008) & (test_years <= 2014)
            bull      = (test_years >= 2015) & (test_years <= 2019)
            covid     = test_years >= 2020

            dacc_crisis = np.mean(np.sign(y_test[crisis]) == np.sign(y_pred[crisis]))*100 if crisis.sum()>5 else np.nan
            dacc_bull   = np.mean(np.sign(y_test[bull])   == np.sign(y_pred[bull])  )*100 if bull.sum()>5   else np.nan
            dacc_covid  = np.mean(np.sign(y_test[covid])  == np.sign(y_pred[covid]) )*100 if covid.sum()>5  else np.nan

            all_results.append({
                "horizon": horizon,
                "test_city": test_city,
                "model": model_name,
                "n_train": len(train_all),
                "n_test": len(test_d),
                "R²": round(r2, 3),
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "Dir.Acc%": round(dacc, 1),
                "Dir.Acc_Crisis%": round(dacc_crisis, 1) if not np.isnan(dacc_crisis) else None,
                "Dir.Acc_Bull%":   round(dacc_bull,   1) if not np.isnan(dacc_bull)   else None,
                "Dir.Acc_Covid%":  round(dacc_covid,  1) if not np.isnan(dacc_covid)  else None,
            })

        # Print best model for this city/horizon
        city_res = [r for r in all_results if r["horizon"]==horizon and r["test_city"]==test_city]
        best = max(city_res, key=lambda x: x["R²"])
        print(f"  Test={test_city:15s}  Best={best['model']:12s}  "
              f"R²={best['R²']:6.3f}  MAE={best['MAE']:.2f}  "
              f"Dir.Acc={best['Dir.Acc%']:.0f}%  "
              f"Crisis={best['Dir.Acc_Crisis%']}%  "
              f"Bull={best['Dir.Acc_Bull%']}%  "
              f"Covid={best['Dir.Acc_Covid%']}%")

# ── 5. Summary across all cities ─────────────────────────────────────────
print("\n\n" + "=" * 70)
print("SUMMARY — Average across all test cities")
print("=" * 70)
res_df = pd.DataFrame(all_results)

summary = (
    res_df.groupby(["horizon", "model"])[["R²","MAE","Dir.Acc%",
                                          "Dir.Acc_Crisis%","Dir.Acc_Bull%","Dir.Acc_Covid%"]]
    .mean().round(3)
)
print(summary.to_string())

# ── 6. Best model per horizon ─────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("BEST MODEL PER HORIZON (highest avg R²)")
print("=" * 70)
for h in HORIZONS:
    sub = res_df[res_df["horizon"] == h]
    avg = sub.groupby("model")["R²"].mean()
    best_model = avg.idxmax()
    best_r2    = avg.max()
    best_mae   = sub[sub["model"]==best_model]["MAE"].mean()
    best_dacc  = sub[sub["model"]==best_model]["Dir.Acc%"].mean()
    best_crisis= sub[sub["model"]==best_model]["Dir.Acc_Crisis%"].mean()
    print(f"  {h:2d}-month:  {best_model:12s}  "
          f"R²={best_r2:.3f}  MAE={best_mae:.2f}%  "
          f"Dir.Acc={best_dacc:.0f}%  "
          f"Crisis dir.acc={best_crisis:.0f}%")

# ── 7. Feature importance (GradBoost on all cities) ──────────────────────
print("\n\n" + "=" * 70)
print("FEATURE IMPORTANCE — GradBoost, 6-month horizon, all cities pooled")
print("=" * 70)
all_data, feat_cols, target_col = build_features(df.copy(), 6)
all_data = all_data.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols + [target_col])
gb_all = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                    learning_rate=0.05, random_state=42)
gb_all.fit(all_data[feat_cols], all_data[target_col])
imp = pd.Series(gb_all.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(imp.head(10).to_string())

# Save
res_df.to_csv(DATA / "direct_forecast_results.csv", index=False)
print(f"\nSaved → data/direct_forecast_results.csv")
