"""
Save trained regression and classification models to disk
so they can be loaded by the monthly pipeline.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import random
import joblib
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DATA    = Path(__file__).parent.parent / "data"
MODELS  = Path(__file__).parent.parent / "nova" / "models"
MODELS.mkdir(parents=True, exist_ok=True)
SEED    = 42
random.seed(SEED); np.random.seed(SEED)

merged = pd.read_csv(DATA / "full_merged.csv", parse_dates=["month"])
merged = merged.replace([np.inf, -np.inf], np.nan)

SIG_COLS  = ["review_yoy","rpr_yoy","checkin_yoy","review_mom",
             "avg_stars","avg_price_tier","pct_budget","pct_upscale"]
FEAT_COLS = [f"{s}_{l}" for s in SIG_COLS for l in ["t0","t1","t3"]
             if f"{s}_{l}" in merged.columns]

# Use ALL cities for final model (no train/test split — max data)
for h in [1, 3, 6, 12]:
    col = f"zhvi_{h}m_fwd"
    if col not in merged.columns:
        merged[col] = merged.groupby("city")["zhvi"].transform(
            lambda x, hh=h: x.pct_change(hh).shift(-hh).mul(100))
    merged[f"dir_{h}m"] = (merged[col] > 0).astype(int)

print("Training final models on ALL 40 cities...")
for h in [1, 3, 6, 12]:
    reg_target = f"zhvi_{h}m_fwd"
    cls_target = f"dir_{h}m"
    df = merged.dropna(subset=FEAT_COLS + [reg_target, cls_target])
    X  = df[FEAT_COLS].values
    y_reg = df[reg_target].values
    y_cls = df[cls_target].values

    # Sample weights for class imbalance
    ratio = (y_cls == 1).sum() / max((y_cls == 0).sum(), 1)
    sw    = np.where(y_cls == 0, ratio, 1.0)

    reg = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, random_state=SEED)
    reg.fit(X, y_reg)
    joblib.dump(reg, MODELS / f"reg_{h}m.pkl")

    cls = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, random_state=SEED)
    cls.fit(X, y_cls, sample_weight=sw)
    joblib.dump(cls, MODELS / f"cls_{h}m.pkl")

    print(f"  Saved reg_{h}m.pkl + cls_{h}m.pkl  "
          f"(trained on {len(df):,} rows)")

# Save feature column names (critical — model expects exact column order)
joblib.dump(FEAT_COLS, MODELS / "feature_cols.pkl")
print(f"\nSaved feature_cols.pkl: {len(FEAT_COLS)} features")
print(f"\nAll models saved to {MODELS}")
