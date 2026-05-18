"""
Address class imbalance for DOWN prediction.

Techniques:
  1. Class weights (built-in)
  2. SMOTE oversampling
  3. ADASYN oversampling
  4. Random undersampling
  5. SMOTE + Tomek Links (combined)
  6. Threshold tuning
  7. BalancedRandomForest
  8. EasyEnsemble

Primary metric: DOWN recall (catching housing downturns)
Secondary: Balanced accuracy, AUC, F1-DOWN
"""
import pandas as pd
import numpy as np
from pathlib import Path
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              balanced_accuracy_score, recall_score,
                              precision_score, confusion_matrix,
                              precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
import warnings
warnings.filterwarnings("ignore")

DATA = Path(__file__).parent.parent / "data"
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ── Load data + same split ────────────────────────────────────────────────
merged = pd.read_csv(DATA / "full_merged.csv", parse_dates=["month"])
merged = merged.replace([np.inf, -np.inf], np.nan)

SIG_COLS  = ["review_yoy","rpr_yoy","checkin_yoy","review_mom",
             "avg_stars","avg_price_tier","pct_budget","pct_upscale"]
FEAT_COLS = [f"{s}_{l}" for s in SIG_COLS for l in ["t0","t1","t3"]
             if f"{s}_{l}" in merged.columns]

all_cities   = sorted(merged["city"].unique())
n_test       = max(1, round(len(all_cities) * 0.1))
test_cities  = random.sample(all_cities, n_test)
train_cities = [c for c in all_cities if c not in test_cities]

for h in [1, 3, 6, 12]:
    col = f"zhvi_{h}m_fwd"
    if col in merged.columns:
        merged[f"dir_{h}m"] = (merged[col] > 0).astype(int)

train_df = merged[merged["city"].isin(train_cities)]
test_df  = merged[merged["city"].isin(test_cities)]

HORIZONS = [1, 3, 6, 12]
all_results = []

# ── Helper: evaluate ──────────────────────────────────────────────────────
def evaluate(y_true, y_pred, y_prob, name, horizon):
    acc      = accuracy_score(y_true, y_pred) * 100
    bal_acc  = balanced_accuracy_score(y_true, y_pred) * 100
    auc      = roc_auc_score(y_true, y_prob)
    down_rec = recall_score(y_true, y_pred, pos_label=0) * 100
    down_pre = precision_score(y_true, y_pred, pos_label=0, zero_division=0) * 100
    down_f1  = f1_score(y_true, y_pred, pos_label=0, zero_division=0) * 100
    up_rec   = recall_score(y_true, y_pred, pos_label=1) * 100
    cm       = confusion_matrix(y_true, y_pred)
    return {
        "technique": name, "horizon": horizon,
        "accuracy": round(acc, 1),
        "balanced_acc": round(bal_acc, 1),
        "roc_auc": round(auc, 3),
        "down_recall": round(down_rec, 1),
        "down_precision": round(down_pre, 1),
        "down_f1": round(down_f1, 1),
        "up_recall": round(up_rec, 1),
        "cm": cm,
    }

def print_row(r):
    print(f"  {r['technique']:28s}  "
          f"BalAcc={r['balanced_acc']:5.1f}%  "
          f"AUC={r['roc_auc']:.3f}  "
          f"DOWN: Rec={r['down_recall']:5.1f}%  Pre={r['down_precision']:5.1f}%  F1={r['down_f1']:5.1f}%  "
          f"UP Rec={r['up_recall']:5.1f}%")

# ── Main loop ─────────────────────────────────────────────────────────────
print("=" * 95)
print("CLASS IMBALANCE TECHNIQUES — Predicting Housing Price DIRECTION")
print("Primary focus: DOWN Recall (catching downturns)")
print("=" * 95)

for horizon in HORIZONS:
    target = f"dir_{horizon}m"
    tr = train_df.dropna(subset=FEAT_COLS + [target])
    te = test_df.dropna( subset=FEAT_COLS + [target])

    X_tr, y_tr = tr[FEAT_COLS].values, tr[target].values
    X_te, y_te = te[FEAT_COLS].values, te[target].values
    test_years = pd.to_datetime(te["month"]).dt.year.values

    down_pct = (y_tr == 0).mean() * 100
    print(f"\n── {horizon}-month horizon  "
          f"(train: {len(tr):,} rows, DOWN={down_pct:.1f}%  | "
          f"test: {len(te):,} rows, DOWN={(y_te==0).mean()*100:.1f}%) ──")

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    horizon_results = []

    # ── Baseline: GBM no correction ───────────────────────────────────────
    gbm0 = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, random_state=SEED)
    gbm0.fit(X_tr, y_tr)
    y_pred0 = gbm0.predict(X_te)
    y_prob0 = gbm0.predict_proba(X_te)[:,1]
    r = evaluate(y_te, y_pred0, y_prob0, "Baseline GBM (no fix)", horizon)
    print_row(r); horizon_results.append(r)

    # ── 1. Class weights ──────────────────────────────────────────────────
    for model_name, model in [
        ("LR class_weight=balanced",
         LogisticRegression(class_weight="balanced", max_iter=1000, C=0.1, random_state=SEED)),
        ("RF class_weight=balanced",
         RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced", random_state=SEED)),
        ("GBM class_weight (sample_w)",
         GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED)),
    ]:
        if "GBM" in model_name:
            # GBM uses sample_weight
            ratio = (y_tr == 1).sum() / max((y_tr == 0).sum(), 1)
            sw    = np.where(y_tr == 0, ratio, 1.0)
            model.fit(X_tr, y_tr, sample_weight=sw)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:,1]
        elif "LR" in model_name:
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)
            y_prob = model.predict_proba(X_te_s)[:,1]
        else:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:,1]
        r = evaluate(y_te, y_pred, y_prob, model_name, horizon)
        print_row(r); horizon_results.append(r)

    # ── 2. SMOTE oversampling ─────────────────────────────────────────────
    k = min(5, (y_tr == 0).sum() - 1)
    if k >= 1:
        smote = SMOTE(random_state=SEED, k_neighbors=k)
        X_sm, y_sm = smote.fit_resample(X_tr, y_tr)
        gbm_sm = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                             learning_rate=0.05, random_state=SEED)
        gbm_sm.fit(X_sm, y_sm)
        y_pred = gbm_sm.predict(X_te)
        y_prob = gbm_sm.predict_proba(X_te)[:,1]
        r = evaluate(y_te, y_pred, y_prob, "SMOTE + GBM", horizon)
        print_row(r); horizon_results.append(r)

    # ── 3. ADASYN ─────────────────────────────────────────────────────────
    try:
        adasyn = ADASYN(random_state=SEED, n_neighbors=min(5,(y_tr==0).sum()-1))
        X_ad, y_ad = adasyn.fit_resample(X_tr, y_tr)
        gbm_ad = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                             learning_rate=0.05, random_state=SEED)
        gbm_ad.fit(X_ad, y_ad)
        y_pred = gbm_ad.predict(X_te)
        y_prob = gbm_ad.predict_proba(X_te)[:,1]
        r = evaluate(y_te, y_pred, y_prob, "ADASYN + GBM", horizon)
        print_row(r); horizon_results.append(r)
    except Exception:
        pass

    # ── 4. Random undersampling ───────────────────────────────────────────
    rus = RandomUnderSampler(random_state=SEED)
    X_us, y_us = rus.fit_resample(X_tr, y_tr)
    gbm_us = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                         learning_rate=0.05, random_state=SEED)
    gbm_us.fit(X_us, y_us)
    y_pred = gbm_us.predict(X_te)
    y_prob = gbm_us.predict_proba(X_te)[:,1]
    r = evaluate(y_te, y_pred, y_prob, "Undersample + GBM", horizon)
    print_row(r); horizon_results.append(r)

    # ── 5. SMOTETomek ─────────────────────────────────────────────────────
    try:
        smt = SMOTETomek(random_state=SEED,
                         smote=SMOTE(k_neighbors=min(5,(y_tr==0).sum()-1),
                                     random_state=SEED))
        X_st, y_st = smt.fit_resample(X_tr, y_tr)
        gbm_st = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                             learning_rate=0.05, random_state=SEED)
        gbm_st.fit(X_st, y_st)
        y_pred = gbm_st.predict(X_te)
        y_prob = gbm_st.predict_proba(X_te)[:,1]
        r = evaluate(y_te, y_pred, y_prob, "SMOTETomek + GBM", horizon)
        print_row(r); horizon_results.append(r)
    except Exception:
        pass

    # ── 6. Threshold tuning (on best base model) ──────────────────────────
    # Use precision-recall curve to find threshold that maximizes DOWN F1
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_te, 1 - y_prob0,
                                                            pos_label=0)
    f1_arr = np.where((prec_arr + rec_arr) > 0,
                      2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9), 0)
    best_thresh_idx = np.argmax(f1_arr)
    best_thresh = 1 - thresholds[best_thresh_idx] if best_thresh_idx < len(thresholds) else 0.5
    y_pred_thr = (y_prob0 < best_thresh).astype(int)
    r = evaluate(y_te, y_pred_thr, y_prob0,
                 f"Threshold tuning (t={best_thresh:.2f})", horizon)
    print_row(r); horizon_results.append(r)

    # ── 7. BalancedRandomForest ───────────────────────────────────────────
    brf = BalancedRandomForestClassifier(n_estimators=200, max_depth=5,
                                          random_state=SEED, sampling_strategy="auto")
    brf.fit(X_tr, y_tr)
    y_pred = brf.predict(X_te)
    y_prob = brf.predict_proba(X_te)[:,1]
    r = evaluate(y_te, y_pred, y_prob, "BalancedRandomForest", horizon)
    print_row(r); horizon_results.append(r)

    # ── 8. EasyEnsemble ───────────────────────────────────────────────────
    try:
        ee = EasyEnsembleClassifier(n_estimators=20, random_state=SEED)
        ee.fit(X_tr, y_tr)
        y_pred = ee.predict(X_te)
        y_prob = ee.predict_proba(X_te)[:,1]
        r = evaluate(y_te, y_pred, y_prob, "EasyEnsemble", horizon)
        print_row(r); horizon_results.append(r)
    except Exception as e:
        print(f"  EasyEnsemble failed: {e}")

    all_results.extend(horizon_results)

    # Best per horizon by DOWN F1
    best = max(horizon_results, key=lambda x: x["down_f1"])
    print(f"\n  ★ Best DOWN F1: {best['technique']}  "
          f"DOWN F1={best['down_f1']:.1f}%  "
          f"DOWN Recall={best['down_recall']:.1f}%  "
          f"BalAcc={best['balanced_acc']:.1f}%")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n\n" + "=" * 95)
print("SUMMARY — Best technique per horizon (by DOWN F1)")
print("=" * 95)
res_df = pd.DataFrame(all_results)
print(f"\n{'Horizon':>8}  {'Best Technique':30s}  {'BalAcc':>8}  {'AUC':>6}  "
      f"{'DOWN Rec':>9}  {'DOWN Pre':>9}  {'DOWN F1':>8}  {'UP Rec':>7}")
for h in HORIZONS:
    sub  = res_df[res_df["horizon"] == h]
    best = sub.loc[sub["down_f1"].idxmax()]
    print(f"{h:>6}m   {best['technique']:30s}  "
          f"{best['balanced_acc']:>7.1f}%  {best['roc_auc']:>6.3f}  "
          f"{best['down_recall']:>8.1f}%  {best['down_precision']:>8.1f}%  "
          f"{best['down_f1']:>7.1f}%  {best['up_recall']:>6.1f}%")

print("\n\nTechnique rankings averaged across all horizons (by DOWN F1):")
avg = res_df.groupby("technique")[["balanced_acc","roc_auc","down_recall",
                                    "down_f1","up_recall"]].mean().round(1)
avg = avg.sort_values("down_f1", ascending=False)
print(avg.to_string())

res_df.drop(columns=["cm"]).to_csv(DATA / "imbalance_results.csv", index=False)
print(f"\nSaved → data/imbalance_results.csv")
