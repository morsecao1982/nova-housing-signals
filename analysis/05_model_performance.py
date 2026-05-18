"""
Detailed model performance analysis.
Compares: baseline, linear regression, and pre-COVID validation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DATA = Path(__file__).parent.parent / "data"

# ── Load merged data ──────────────────────────────────────────────────────
df = pd.read_csv(DATA / "merged_analysis.csv", parse_dates=["month"])
df = df.replace([np.inf, -np.inf], np.nan)

CITIES = df["city"].unique()
TARGET = "zhvi_6m_fwd"

FEATURES_V1 = ["review_yoy_growth", "checkin_yoy_growth"]

FEATURES_V2 = [
    "review_volume",
    "reviews_per_restaurant",
    "checkin_volume",
    "avg_stars",
    "review_volume_3m_avg",
]

print("=" * 70)
print("MODEL PERFORMANCE REPORT")
print("=" * 70)

# ── Helper ────────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    direction = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    return {"model": name, "R²": round(r2,3), "MAE": round(mae,2),
            "RMSE": round(rmse,2), "Dir.Acc%": round(direction,1)}

results = []

for city in sorted(CITIES):
    city_df = df[df["city"] == city].sort_values("month").copy()

    # ── Split periods ─────────────────────────────────────────────────────
    pre_covid  = city_df[city_df["month"] < "2020-01-01"]
    post_covid = city_df[city_df["month"] >= "2020-01-01"]
    train      = pre_covid.copy()

    print(f"\n{'─'*70}")
    print(f"  {city.upper()}")
    print(f"  Train (pre-COVID): {len(train)} months | "
          f"Test (COVID+): {len(post_covid)} months")
    print(f"{'─'*70}")

    for features, fname in [(FEATURES_V1, "YoY Growth features"),
                             (FEATURES_V2, "Volume features")]:
        tr = train.dropna(subset=features + [TARGET])
        te = post_covid.dropna(subset=features + [TARGET])

        if len(tr) < 20 or len(te) < 6:
            continue

        X_tr, y_tr = tr[features].values, tr[TARGET].values
        X_te, y_te = te[features].values, te[TARGET].values

        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        city_results = []

        # 1. Naive baseline: always predict train mean
        naive_pred_tr = np.full_like(y_tr, y_tr.mean())
        naive_pred_te = np.full_like(y_te, y_tr.mean())
        city_results.append({**evaluate("Baseline (mean)", y_tr, naive_pred_tr), "split": "train"})
        city_results.append({**evaluate("Baseline (mean)", y_te, naive_pred_te), "split": "test"})

        # 2. Linear Regression
        lr = LinearRegression()
        lr.fit(X_tr_s, y_tr)
        city_results.append({**evaluate("Linear Regression", y_tr, lr.predict(X_tr_s)), "split": "train"})
        city_results.append({**evaluate("Linear Regression", y_te, lr.predict(X_te_s)), "split": "test"})

        # 3. Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr_s, y_tr)
        city_results.append({**evaluate("Ridge Regression", y_tr, ridge.predict(X_tr_s)), "split": "train"})
        city_results.append({**evaluate("Ridge Regression", y_te, ridge.predict(X_te_s)), "split": "test"})

        # 4. Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        gb.fit(X_tr, y_tr)
        city_results.append({**evaluate("Gradient Boosting", y_tr, gb.predict(X_tr)), "split": "train"})
        city_results.append({**evaluate("Gradient Boosting", y_te, gb.predict(X_te)), "split": "test"})

        print(f"\n  Features: {fname}")
        cr_df = pd.DataFrame(city_results)
        pivot = cr_df.pivot_table(
            index="model", columns="split",
            values=["R²","MAE","Dir.Acc%"], aggfunc="first"
        )
        print(pivot.to_string())

        # Store for summary
        for r in city_results:
            results.append({"city": city, "features": fname, **r})

    # ── Pre-COVID only cross-validation (walk-forward) ───────────────────
    print(f"\n  Pre-COVID Walk-Forward Validation (2010–2019):")
    cv_data = pre_covid.dropna(subset=FEATURES_V2 + [TARGET])
    cv_data = cv_data[cv_data["month"] >= "2010-01-01"]

    if len(cv_data) >= 36:
        # Use first 60% to train, predict last 40%
        split = int(len(cv_data) * 0.6)
        tr_cv = cv_data.iloc[:split]
        te_cv = cv_data.iloc[split:]

        X_tr_cv = tr_cv[FEATURES_V2].values
        y_tr_cv = tr_cv[TARGET].values
        X_te_cv = te_cv[FEATURES_V2].values
        y_te_cv = te_cv[TARGET].values

        sc = StandardScaler()
        X_tr_cv_s = sc.fit_transform(X_tr_cv)
        X_te_cv_s = sc.transform(X_te_cv)

        lr_cv = LinearRegression().fit(X_tr_cv_s, y_tr_cv)
        gb_cv = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        gb_cv.fit(X_tr_cv, y_tr_cv)

        lr_res  = evaluate("Linear (pre-COVID cv)", y_te_cv, lr_cv.predict(X_te_cv_s))
        gb_res  = evaluate("GradBoost (pre-COVID cv)", y_te_cv, gb_cv.predict(X_te_cv))

        for r in [lr_res, gb_res]:
            print(f"    {r['model']:35s} R²={r['R²']:6.3f}  MAE={r['MAE']:5.2f}  Dir.Acc={r['Dir.Acc%']}%")

# ── Summary across all cities ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY: Average test performance across all cities")
print(f"{'='*70}")
summary_df = pd.DataFrame(results)
summary = (
    summary_df[summary_df["split"] == "test"]
    .groupby(["features", "model"])[["R²","MAE","Dir.Acc%"]]
    .mean()
    .round(3)
)
print(summary.to_string())

print(f"""
{'='*70}
INTERPRETATION
{'='*70}
R²  > 0.5  : Strong predictive power
R²  > 0.2  : Moderate predictive power
R²  < 0    : Worse than predicting the mean (model is broken)

MAE        : Average prediction error in percentage points
             e.g. MAE=3 means predictions are off by ±3% on average

Dir.Acc%   : % of time the model correctly predicts direction
             (up vs down) — most useful for investment decisions
             Baseline = ~50% (coin flip)
             >60%     = useful signal
             >70%     = strong signal
""")
