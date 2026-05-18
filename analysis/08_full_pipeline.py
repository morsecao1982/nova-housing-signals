"""
Full pipeline for all 40 cities — 9:1 city-based train/test split.
Steps:
  1. Build city list (200+ restaurants, Zillow match)
  2. Build city-level ZHVI from ZIP-level Zillow data
  3. Stream Yelp reviews + checkins for all 40 cities
  4. Build monthly signals
  5. Train direct forecast model (1/3/6/12-month horizons)
  6. 9:1 random city split — report performance
"""
import json, random
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DATA    = Path(__file__).parent.parent / "data"
YELP    = DATA / "Yelp JSON"
SEED    = 42
random.seed(SEED)
np.random.seed(SEED)

# ── 1. Build city list ────────────────────────────────────────────────────
print("Step 1: Identifying cities...")
businesses_raw = []
with open(YELP / "yelp_academic_dataset_business.json", encoding="utf-8") as f:
    for line in f:
        b = json.loads(line)
        state = b.get("state","")
        if state in ("AB","BC","ON","QC"):
            continue
        cats = b.get("categories") or ""
        if not isinstance(cats, str): continue
        if not any(c.strip() in {
            "Restaurants","Food","Bars","Fast Food","Pizza","Burgers",
            "Sandwiches","Mexican","Chinese","Italian","American (New)",
            "American (Traditional)","Cafes","Breakfast & Brunch",
        } for c in cats.split(",")): continue
        attrs = b.get("attributes") or {}
        price = attrs.get("RestaurantsPriceRange2")
        try: price = int(price)
        except: price = None
        businesses_raw.append({
            "business_id": b["business_id"],
            "city": b.get("city",""), "state": state,
            "stars": b.get("stars"), "is_open": b.get("is_open",1),
            "price_tier": price,
        })

biz_df    = pd.DataFrame(businesses_raw)
city_cnt  = biz_df.groupby(["city","state"]).size().reset_index(name="n")
city_cnt  = city_cnt[city_cnt["n"] >= 200]

# Load Zillow ZIP, keep cities that exist there
zip_df = pd.read_csv(DATA / "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")
valid_cities = []
for _, row in city_cnt.iterrows():
    mask = (zip_df["City"].str.lower() == row["city"].lower()) & (zip_df["State"] == row["state"])
    if mask.sum() > 0:
        valid_cities.append((row["city"], row["state"], row["n"]))

print(f"  Valid cities: {len(valid_cities)}")
CITY_STATES = {(c, s) for c, s, _ in valid_cities}
CITIES      = sorted({c for c, s, _ in valid_cities})
city_state_map = {c: s for c, s, _ in valid_cities}

# Filter biz_df to valid cities
biz_df = biz_df[biz_df.apply(lambda r: (r["city"], r["state"]) in CITY_STATES, axis=1)]
biz_ids   = set(biz_df["business_id"])
city_map  = biz_df.set_index("business_id")["city"].to_dict()
price_map = biz_df.set_index("business_id")["price_tier"].to_dict()
print(f"  Total restaurants: {len(biz_df):,}")

# ── 2. Build city-level ZHVI from ZIP data ────────────────────────────────
print("\nStep 2: Building city-level housing data from Zillow ZIPs...")
date_cols = [c for c in zip_df.columns if c.startswith("20")]
housing_rows = []
for city, state in CITY_STATES:
    mask = (zip_df["City"].str.lower() == city.lower()) & (zip_df["State"] == state)
    city_zips = zip_df[mask][date_cols]
    if city_zips.empty: continue
    # Average ZHVI across all ZIPs in this city
    avg_zhvi = city_zips.mean(axis=0)
    for date, val in avg_zhvi.items():
        if pd.isna(val): continue
        housing_rows.append({"city": city, "state": state,
                              "month": pd.to_datetime(date).to_period("M").to_timestamp(),
                              "zhvi": val})

housing_df = pd.DataFrame(housing_rows)
print(f"  Housing rows: {len(housing_df):,}")
print(f"  Date range: {housing_df['month'].min().date()} → {housing_df['month'].max().date()}")

# Compute forward targets
housing_df = housing_df.sort_values(["city","month"])
for city in CITIES:
    m = housing_df["city"] == city
    for h in [1, 3, 6, 12]:
        housing_df.loc[m, f"zhvi_{h}m_fwd"] = (
            housing_df.loc[m, "zhvi"].pct_change(h).shift(-h).mul(100)
        )
    housing_df.loc[m, "zhvi_yoy"] = housing_df.loc[m, "zhvi"].pct_change(12).mul(100)

# ── 3. Stream Yelp reviews ────────────────────────────────────────────────
print("\nStep 3: Streaming reviews (5GB)...")
review_counts = defaultdict(lambda: defaultdict(int))
review_stars  = defaultdict(lambda: defaultdict(list))

with open(YELP / "yelp_academic_dataset_review.json", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reviews"):
        r = json.loads(line)
        bid = r.get("business_id")
        if bid not in biz_ids: continue
        city  = city_map[bid]
        month = r["date"][:7]
        review_counts[city][month] += 1
        review_stars[city][month].append(r.get("stars", 0))

# ── 4. Stream checkins ────────────────────────────────────────────────────
print("\nStep 4: Streaming checkins...")
checkin_counts = defaultdict(lambda: defaultdict(int))

with open(YELP / "yelp_academic_dataset_checkin.json", encoding="utf-8") as f:
    for line in tqdm(f, desc="Checkins"):
        c = json.loads(line)
        bid = c.get("business_id")
        if bid not in biz_ids: continue
        city = city_map[bid]
        for ts in (c.get("date") or "").split(","):
            ts = ts.strip()
            if len(ts) >= 7:
                checkin_counts[city][ts[:7]] += 1

# ── 5. Assemble monthly signals ───────────────────────────────────────────
print("\nStep 5: Building signals...")
rows = []
for city in CITIES:
    n_restaurants = len(biz_df[biz_df["city"] == city])
    price_tiers   = biz_df[biz_df["city"] == city]["price_tier"].dropna()
    avg_price     = price_tiers.mean() if len(price_tiers) > 0 else np.nan
    pct_budget    = (price_tiers == 1).mean() if len(price_tiers) > 0 else np.nan
    pct_upscale   = (price_tiers >= 3).mean() if len(price_tiers) > 0 else np.nan

    for month in sorted(review_counts[city].keys()):
        yr = int(month[:4])
        if yr < 2005 or yr > 2022: continue
        stars_list = review_stars[city][month]
        rows.append({
            "city":          city,
            "state":         city_state_map[city],
            "month":         month,
            "review_volume": review_counts[city][month],
            "checkin_volume":checkin_counts[city].get(month, 0),
            "avg_stars":     np.mean(stars_list) if stars_list else np.nan,
            "n_restaurants": n_restaurants,
            "reviews_per_restaurant": review_counts[city][month] / n_restaurants,
            "avg_price_tier": avg_price,
            "pct_budget":    pct_budget,
            "pct_upscale":   pct_upscale,
        })

signals = pd.DataFrame(rows)
signals["month"] = pd.to_datetime(signals["month"]).dt.to_period("M").dt.to_timestamp()
signals = signals.sort_values(["city","month"])

# YoY and MoM features
for city in CITIES:
    m = signals["city"] == city
    signals.loc[m, "review_yoy"]  = signals.loc[m, "review_volume"].pct_change(12).mul(100)
    signals.loc[m, "rpr_yoy"]     = signals.loc[m, "reviews_per_restaurant"].pct_change(12).mul(100)
    signals.loc[m, "checkin_yoy"] = signals.loc[m, "checkin_volume"].pct_change(12).mul(100)
    signals.loc[m, "review_mom"]  = signals.loc[m, "review_volume"].pct_change(1).mul(100)

# ── 6. Merge signals + housing ────────────────────────────────────────────
print("\nStep 6: Merging...")
merged = pd.merge(signals, housing_df, on=["city","month"], how="inner")
merged = merged.replace([np.inf, -np.inf], np.nan)
print(f"  Merged rows: {len(merged):,}")
print(f"  Cities with data: {merged['city'].nunique()}")

# ── 7. Build feature matrix ───────────────────────────────────────────────
SIG_COLS  = ["review_yoy","rpr_yoy","checkin_yoy","review_mom","avg_stars",
             "avg_price_tier","pct_budget","pct_upscale"]
FEAT_COLS = []
for sig in SIG_COLS:
    if sig not in merged.columns: continue
    merged[f"{sig}_t0"] = merged.groupby("city")[sig].transform(lambda x: x)
    merged[f"{sig}_t1"] = merged.groupby("city")[sig].transform(lambda x: x.shift(1))
    merged[f"{sig}_t3"] = merged.groupby("city")[sig].transform(lambda x: x.shift(3))
    FEAT_COLS += [f"{sig}_t0", f"{sig}_t1", f"{sig}_t3"]

# ── 8. Train/test split 9:1 by city ──────────────────────────────────────
all_cities = sorted(merged["city"].unique())
n_test     = max(1, round(len(all_cities) * 0.1))
test_cities = random.sample(all_cities, n_test)
train_cities = [c for c in all_cities if c not in test_cities]

print(f"\nStep 7: 9:1 City Split")
print(f"  Train cities ({len(train_cities)}): {', '.join(sorted(train_cities))}")
print(f"  Test  cities ({len(test_cities)}):  {', '.join(sorted(test_cities))}")

# ── 9. Train and evaluate ─────────────────────────────────────────────────
print("\n" + "="*70)
print("RESULTS — Direct Forecast, 9:1 City Split")
print("="*70)

HORIZONS = [1, 3, 6, 12]
MODELS   = {
    "Ridge":      lambda: Ridge(alpha=1.0),
    "RandForest": lambda: RandomForestRegressor(n_estimators=200, max_depth=5, random_state=SEED),
    "GradBoost":  lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                     learning_rate=0.05, random_state=SEED),
}

results = []
train_df = merged[merged["city"].isin(train_cities)]
test_df  = merged[merged["city"].isin(test_cities)]

for horizon in HORIZONS:
    target = f"zhvi_{horizon}m_fwd"
    tr = train_df.dropna(subset=FEAT_COLS + [target])
    te = test_df.dropna(subset=FEAT_COLS + [target])

    if len(tr) < 100 or len(te) < 20:
        print(f"  {horizon}m: insufficient data")
        continue

    X_tr, y_tr = tr[FEAT_COLS].values, tr[target].values
    X_te, y_te = te[FEAT_COLS].values, te[target].values
    test_years = pd.to_datetime(te["month"]).dt.year.values

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    print(f"\n  {horizon}-month horizon  "
          f"(train: {len(tr):,} rows / {len(train_cities)} cities | "
          f"test: {len(te):,} rows / {len(test_cities)} cities)")

    for mname, mfunc in MODELS.items():
        model = mfunc()
        if mname == "Ridge":
            model.fit(X_tr_s, y_tr); y_pred = model.predict(X_te_s)
        else:
            model.fit(X_tr, y_tr);   y_pred = model.predict(X_te)

        r2   = r2_score(y_te, y_pred)
        mae  = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(np.mean((y_te - y_pred)**2))
        dacc = np.mean(np.sign(y_te) == np.sign(y_pred)) * 100
        corr = np.corrcoef(y_te, y_pred)[0,1] if len(y_te) > 2 else np.nan

        # Period breakdown
        crisis = (test_years >= 2008) & (test_years <= 2014)
        bull   = (test_years >= 2015) & (test_years <= 2019)
        covid  = test_years >= 2020

        def dacc_period(mask):
            return round(np.mean(np.sign(y_te[mask]) == np.sign(y_pred[mask]))*100, 1) \
                   if mask.sum() > 5 else None

        print(f"    {mname:12s}  R²={r2:6.3f}  MAE={mae:.2f}%  RMSE={rmse:.2f}%  "
              f"Dir.Acc={dacc:.0f}%  "
              f"Crisis={dacc_period(crisis)}%  "
              f"Bull={dacc_period(bull)}%  "
              f"Covid={dacc_period(covid)}%")

        results.append({
            "horizon": horizon, "model": mname,
            "R²": round(r2,3), "MAE": round(mae,2), "RMSE": round(rmse,2),
            "Dir.Acc%": round(dacc,1),
            "Dir.Acc_Crisis%": dacc_period(crisis),
            "Dir.Acc_Bull%":   dacc_period(bull),
            "Dir.Acc_Covid%":  dacc_period(covid),
            "n_train_rows": len(tr), "n_test_rows": len(te),
        })

# ── 10. Summary ───────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print("BEST MODEL PER HORIZON")
print("="*70)
res_df = pd.DataFrame(results)
for h in HORIZONS:
    sub = res_df[res_df["horizon"] == h]
    if sub.empty: continue
    best = sub.loc[sub["R²"].idxmax()]
    print(f"  {h:2d}-month  {best['model']:12s}  "
          f"R²={best['R²']:.3f}  MAE={best['MAE']:.2f}%  "
          f"Dir.Acc={best['Dir.Acc%']:.0f}%  "
          f"Crisis={best['Dir.Acc_Crisis%']}%")

# Feature importance
print("\n\n" + "="*70)
print("FEATURE IMPORTANCE — GradBoost, 6-month horizon")
print("="*70)
target = "zhvi_6m_fwd"
full_clean = merged.dropna(subset=FEAT_COLS + [target]).replace([np.inf,-np.inf], np.nan).dropna(subset=FEAT_COLS + [target])
gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED)
gb.fit(full_clean[FEAT_COLS], full_clean[target])
imp = pd.Series(gb.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
print(imp.head(12).to_string())

# Save
res_df.to_csv(DATA / "full_model_results.csv", index=False)
merged.to_csv(DATA / "full_merged.csv", index=False)
print(f"\nSaved → data/full_model_results.csv, data/full_merged.csv")
