"""
Load trained models and generate buy/sell signals for Northern Virginia.
Reads: data/nova_signals.csv
Writes: public/nova_results.json  (read by website)
"""
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

DATA    = Path(__file__).parent.parent / "data"
MODELS  = Path(__file__).parent / "models"
PUBLIC  = Path(__file__).parent.parent / "public"
PUBLIC.mkdir(exist_ok=True)

HORIZONS = [1, 3, 6, 12]

def generate_signal(down_prob, pred_pct):
    up_prob = 1 - down_prob
    if down_prob >= 0.55 and pred_pct < -0.5:  return "STRONG SELL", -2
    elif down_prob >= 0.45 or pred_pct < -0.3: return "SELL",        -1
    elif up_prob  >= 0.75 and pred_pct >= 2.0: return "STRONG BUY",  +2
    elif up_prob  >= 0.60 and pred_pct >= 0.5: return "BUY",         +1
    else:                                       return "HOLD",          0

def run():
    # Load signals
    sig_path = DATA / "nova_signals.csv"
    if not sig_path.exists():
        print("nova_signals.csv not found. Run compute_signals.py first.")
        return

    df       = pd.read_csv(sig_path, parse_dates=["month"])
    df       = df.replace([np.inf, -np.inf], np.nan)
    feat_cols= joblib.load(MODELS / "feature_cols.pkl")

    # Use latest month only
    latest = df["month"].max()
    df_now = df[df["month"] == latest].copy()

    # Check how many features are available (bootstrap vs mature)
    available_feats = [c for c in feat_cols if c in df_now.columns]
    missing_feats   = [c for c in feat_cols if c not in df_now.columns]

    print(f"Latest data: {latest.date()}")
    print(f"Areas: {sorted(df_now['area'].unique())}")
    print(f"Features available: {len(available_feats)}/{len(feat_cols)}")
    if missing_feats:
        print(f"  Missing (will fill 0): {missing_feats[:5]}...")

    # Fill missing features with 0
    for c in missing_feats:
        df_now[c] = 0.0

    X = df_now[feat_cols].fillna(0).values

    results = {
        "generated_at": datetime.now().isoformat(),
        "data_month":   latest.strftime("%Y-%m"),
        "bootstrap":    len(missing_feats) > 0,
        "confidence":   "low" if len(missing_feats) > 10 else
                        "medium" if len(missing_feats) > 0 else "high",
        "areas": {}
    }

    for _, row in df_now.iterrows():
        area = row["area"]
        x    = df_now[df_now["area"] == area][feat_cols].fillna(0).values

        area_signals = {}
        for h in HORIZONS:
            try:
                reg = joblib.load(MODELS / f"reg_{h}m.pkl")
                cls = joblib.load(MODELS / f"cls_{h}m.pkl")
                pred_pct  = float(reg.predict(x)[0])
                pred_prob = cls.predict_proba(x)[0]
                down_prob = float(pred_prob[0])
                signal, score = generate_signal(down_prob, pred_pct)
                area_signals[f"{h}m"] = {
                    "signal":       signal,
                    "score":        score,
                    "pred_pct":     round(pred_pct, 2),
                    "down_prob":    round(down_prob, 3),
                    "up_prob":      round(1 - down_prob, 3),
                }
            except Exception as e:
                area_signals[f"{h}m"] = {"signal": "HOLD", "score": 0, "error": str(e)}

        # Primary signal = 6-month (most reliable)
        primary = area_signals.get("6m", {})

        # Determine region from snapshot data
        region = "Northern Virginia"
        snap_dir = Path(__file__).parent.parent / "data" / "nova_snapshots"
        latest_snap = sorted(snap_dir.glob("*.json"))[-1] if list(snap_dir.glob("*.json")) else None
        if latest_snap:
            import json as _json
            snap = _json.load(open(latest_snap))
            area_rests = snap.get("areas", {}).get(area, [])
            if area_rests:
                region = area_rests[0].get("region", "Northern Virginia")

        results["areas"][area] = {
            "region":         region,
            "primary_signal": primary.get("signal", "HOLD"),
            "primary_score":  primary.get("score", 0),
            "n_restaurants":  int(row.get("n_restaurants", 0)),
            "avg_stars":      round(float(row.get("avg_stars", 0) or 0), 2),
            "avg_price_tier": round(float(row.get("avg_price_tier", 0) or 0), 2),
            "review_mom":     round(float(row.get("review_mom", 0) or 0), 2),
            "review_yoy":     round(float(row.get("review_yoy", 0) or 0), 2),
            "horizons":       area_signals,
        }

    out_path = PUBLIC / "nova_results.json"
    # Replace NaN/Inf with None so the output is valid JSON
    def sanitize(obj):
        if isinstance(obj, dict):   return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):   return [sanitize(v) for v in obj]
        if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
            return None
        return obj

    with open(out_path, "w") as f:
        json.dump(sanitize(results), f, indent=2)

    print(f"\nSignals generated:")
    for area, data in sorted(results["areas"].items()):
        sig = data["primary_signal"]
        pct = data["horizons"].get("6m", {}).get("pred_pct", 0)
        print(f"  {area:20s}  {sig:12s}  6m pred: {pct:+.2f}%")

    print(f"\nSaved → {out_path}")
    print(f"Confidence: {results['confidence']} "
          f"({'bootstrap mode' if results['bootstrap'] else 'full model'})")

if __name__ == "__main__":
    run()
