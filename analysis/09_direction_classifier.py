"""
Binary classification: predict if housing price goes UP or DOWN
over the next 1/3/6/12 months.

Models: Logistic Regression, Random Forest, GBM, SVM, KNN,
        AdaBoost, Extra Trees, Naive Bayes
Evaluation: Accuracy, ROC-AUC, F1, Precision, Recall
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score, confusion_matrix,
                              classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

DATA = Path(__file__).parent.parent / "data"
SEED = 42
np.random.seed(SEED)

# ── 1. Load data ──────────────────────────────────────────────────────────
print("Loading data...")
merged = pd.read_csv(DATA / "full_merged.csv", parse_dates=["month"])
merged = merged.replace([np.inf, -np.inf], np.nan)

# ── 2. Build feature matrix (same as regression model) ───────────────────
SIG_COLS  = ["review_yoy", "rpr_yoy", "checkin_yoy", "review_mom",
             "avg_stars", "avg_price_tier", "pct_budget", "pct_upscale"]
FEAT_COLS = []
for sig in SIG_COLS:
    if sig not in merged.columns: continue
    for lag in ["t0", "t1", "t3"]:
        col = f"{sig}_{lag}"
        if col not in merged.columns:
            merged[col] = merged.groupby("city")[sig].transform(
                lambda x, l=lag: x.shift(int(l[1:])) if l != "t0" else x
            )
        FEAT_COLS.append(col)

FEAT_COLS = [c for c in FEAT_COLS if c in merged.columns]

# ── 3. Same 9:1 city split ────────────────────────────────────────────────
import random
random.seed(SEED)
all_cities  = sorted(merged["city"].unique())
n_test      = max(1, round(len(all_cities) * 0.1))
test_cities = random.sample(all_cities, n_test)
train_cities= [c for c in all_cities if c not in test_cities]

print(f"Train: {len(train_cities)} cities | Test: {len(test_cities)} cities")
print(f"Test cities: {', '.join(sorted(test_cities))}\n")

train_df = merged[merged["city"].isin(train_cities)]
test_df  = merged[merged["city"].isin(test_cities)]

# ── 4. Model definitions ──────────────────────────────────────────────────
MODELS = {
    "Logistic Reg":   LogisticRegression(max_iter=1000, random_state=SEED, C=0.1),
    "Random Forest":  RandomForestClassifier(n_estimators=200, max_depth=5, random_state=SEED),
    "Gradient Boost": GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                  learning_rate=0.05, random_state=SEED),
    "SVM":            CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=SEED)),
    "KNN":            KNeighborsClassifier(n_neighbors=10),
    "AdaBoost":       AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED),
    "Extra Trees":    ExtraTreesClassifier(n_estimators=200, max_depth=5, random_state=SEED),
    "Naive Bayes":    GaussianNB(),
}

HORIZONS = [1, 3, 6, 12]
all_results = []

# ── 5. Train and evaluate ─────────────────────────────────────────────────
print("=" * 75)
print("DIRECTION CLASSIFICATION RESULTS")
print("Target: 1 = price UP, 0 = price DOWN/FLAT over next X months")
print("=" * 75)

for horizon in HORIZONS:
    target_reg = f"zhvi_{horizon}m_fwd"

    # Compute binary targets from regression columns
    target_cls = f"dir_{horizon}m"
    merged[target_cls] = (merged[target_reg] > 0).astype(int)
    train_df = merged[merged["city"].isin(train_cities)]
    test_df  = merged[merged["city"].isin(test_cities)]

    tr = train_df.dropna(subset=FEAT_COLS + [target_cls])
    te = test_df.dropna(subset=FEAT_COLS  + [target_cls])

    X_tr, y_tr = tr[FEAT_COLS].values, tr[target_cls].values
    X_te, y_te = te[FEAT_COLS].values, te[target_cls].values
    test_years  = pd.to_datetime(te["month"]).dt.year.values

    # Class balance
    up_rate_train = y_tr.mean() * 100
    up_rate_test  = y_te.mean() * 100

    print(f"\n── {horizon}-month horizon ──")
    print(f"   Train rows: {len(tr):,} | UP rate: {up_rate_train:.1f}%")
    print(f"   Test  rows: {len(te):,} | UP rate: {up_rate_test:.1f}%")
    print(f"   Baseline (always predict UP): {up_rate_test:.1f}%\n")

    # Period masks
    crisis = (test_years >= 2008) & (test_years <= 2014)
    bull   = (test_years >= 2015) & (test_years <= 2019)
    covid  = test_years >= 2020

    # Scale
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    horizon_results = []

    for mname, model in MODELS.items():
        # Fit
        needs_scale = mname in ("Logistic Reg", "SVM", "KNN", "Naive Bayes")
        if needs_scale:
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)
            y_prob = model.predict_proba(X_te_s)[:, 1]
        else:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]

        acc   = accuracy_score(y_te, y_pred) * 100
        auc   = roc_auc_score(y_te, y_prob)
        f1    = f1_score(y_te, y_pred) * 100
        prec  = precision_score(y_te, y_pred) * 100
        rec   = recall_score(y_te, y_pred) * 100

        # Period accuracy
        def period_acc(mask):
            if mask.sum() < 5: return None
            return round(accuracy_score(y_te[mask], y_pred[mask]) * 100, 1)

        acc_crisis = period_acc(crisis)
        acc_bull   = period_acc(bull)
        acc_covid  = period_acc(covid)

        print(f"   {mname:15s}  Acc={acc:.1f}%  AUC={auc:.3f}  "
              f"F1={f1:.1f}%  Prec={prec:.1f}%  Rec={rec:.1f}%  "
              f"| Crisis={acc_crisis}%  Bull={acc_bull}%  Covid={acc_covid}%")

        horizon_results.append({
            "horizon": horizon, "model": mname,
            "accuracy": round(acc, 1), "roc_auc": round(auc, 3),
            "f1": round(f1, 1), "precision": round(prec, 1), "recall": round(rec, 1),
            "acc_crisis": acc_crisis, "acc_bull": acc_bull, "acc_covid": acc_covid,
            "n_train": len(tr), "n_test": len(te),
            "baseline_acc": round(up_rate_test, 1),
        })
        all_results.append(horizon_results[-1])

    # Best model for this horizon
    best = max(horizon_results, key=lambda x: x["roc_auc"])
    lift = best["accuracy"] - best["baseline_acc"]
    print(f"\n   ★ Best: {best['model']}  "
          f"Acc={best['accuracy']}%  AUC={best['roc_auc']}  "
          f"Lift over baseline=+{lift:.1f}%")

# ── 6. Overall summary ────────────────────────────────────────────────────
print("\n\n" + "=" * 75)
print("SUMMARY — Best model per horizon (by ROC-AUC)")
print("=" * 75)
res_df = pd.DataFrame(all_results)

for h in HORIZONS:
    sub  = res_df[res_df["horizon"] == h]
    best = sub.loc[sub["roc_auc"].idxmax()]
    base = best["baseline_acc"]
    lift = best["accuracy"] - base
    print(f"  {h:2d}-month  {best['model']:15s}  "
          f"Acc={best['accuracy']:.1f}%  AUC={best['roc_auc']:.3f}  "
          f"F1={best['f1']:.1f}%  "
          f"Baseline={base:.1f}%  Lift=+{lift:.1f}%  "
          f"Crisis={best['acc_crisis']}%")

# ── 7. Model comparison table (avg across horizons) ──────────────────────
print("\n\n" + "=" * 75)
print("MODEL COMPARISON — Average across all horizons")
print("=" * 75)
avg = res_df.groupby("model")[["accuracy","roc_auc","f1"]].mean().round(3)
avg["avg_lift"] = (res_df.groupby("model").apply(
    lambda g: (g["accuracy"] - g["baseline_acc"]).mean()
)).round(1)
avg = avg.sort_values("roc_auc", ascending=False)
print(avg.to_string())

# ── 8. Confusion matrix for best overall model ────────────────────────────
print("\n\n" + "=" * 75)
print("CONFUSION MATRIX — Best model, 6-month horizon")
print("=" * 75)
best_6m = res_df[res_df["horizon"] == 6].loc[
    res_df[res_df["horizon"] == 6]["roc_auc"].idxmax()
]
print(f"Model: {best_6m['model']}")

target_cls = "dir_6m"
tr6 = train_df.dropna(subset=FEAT_COLS + [target_cls])
te6 = test_df.dropna(subset=FEAT_COLS  + [target_cls])
X_tr6, y_tr6 = tr6[FEAT_COLS].values, tr6[target_cls].values
X_te6, y_te6 = te6[FEAT_COLS].values, te6[target_cls].values
sc6 = StandardScaler()
X_tr6_s = sc6.fit_transform(X_tr6)
X_te6_s = sc6.transform(X_te6)

best_model_name = best_6m["model"]
best_model_obj  = MODELS[best_model_name]
needs_scale = best_model_name in ("Logistic Reg", "SVM", "KNN", "Naive Bayes")
if needs_scale:
    best_model_obj.fit(X_tr6_s, y_tr6); y_pred6 = best_model_obj.predict(X_te6_s)
else:
    best_model_obj.fit(X_tr6, y_tr6);   y_pred6 = best_model_obj.predict(X_te6)

cm = confusion_matrix(y_te6, y_pred6)
print(f"\n  Actual↓  Pred→    DOWN    UP")
print(f"  DOWN           {cm[0,0]:6d}  {cm[0,1]:6d}")
print(f"  UP             {cm[1,0]:6d}  {cm[1,1]:6d}")
print(f"\n{classification_report(y_te6, y_pred6, target_names=['DOWN','UP'])}")

# Save
res_df.to_csv(DATA / "classifier_results.csv", index=False)
print(f"Saved → data/classifier_results.csv")
