"""
Build monthly restaurant signals per city from Yelp data.
Processes large files (review 5GB, checkin 274MB) via streaming.

Output: data/restaurant_signals.csv
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

DATA  = Path(__file__).parent.parent / "data" / "Yelp JSON"
OUT   = Path(__file__).parent.parent / "data"

CITIES = {
    "Philadelphia": "Philadelphia, PA",
    "Tampa":        "Tampa, FL",
    "Indianapolis": "Indianapolis, IN",
    "Nashville":    "Nashville, TN",
    "New Orleans":  "New Orleans, LA",
}

# ── 1. Load restaurants for our 5 cities ─────────────────────────────────
print("Loading business data...")
businesses = []
with open(DATA / "yelp_academic_dataset_business.json", encoding="utf-8") as f:
    for line in f:
        b = json.loads(line)
        if b.get("city") not in CITIES:
            continue
        cats = b.get("categories") or ""
        if not isinstance(cats, str):
            continue
        # Keep food/restaurant businesses
        if not any(c.strip() in {
            "Restaurants","Food","Bars","Fast Food","Pizza","Burgers",
            "Sandwiches","Mexican","Chinese","Italian","American (New)",
            "American (Traditional)","Cafes","Breakfast & Brunch","Nightlife",
        } for c in cats.split(",")):
            continue
        # Extract price tier
        attrs = b.get("attributes") or {}
        price = attrs.get("RestaurantsPriceRange2")
        try:
            price = int(price)
        except (TypeError, ValueError):
            price = None
        businesses.append({
            "business_id": b["business_id"],
            "city":        b["city"],
            "stars":       b.get("stars"),
            "review_count":b.get("review_count", 0),
            "is_open":     b.get("is_open", 1),
            "price_tier":  price,
            "latitude":    b.get("latitude"),
            "longitude":   b.get("longitude"),
        })

biz_df = pd.DataFrame(businesses)
biz_ids = set(biz_df["business_id"])
city_map = biz_df.set_index("business_id")["city"].to_dict()
price_map = biz_df.set_index("business_id")["price_tier"].to_dict()

print(f"Restaurants in target cities: {len(biz_df):,}")
print(biz_df.groupby("city").size().to_string())
print()

# ── 2. Stream review file → monthly review volume per city ───────────────
print("Streaming review data (5GB — this takes a few minutes)...")
review_counts = defaultdict(lambda: defaultdict(int))   # city → month → count
review_stars  = defaultdict(lambda: defaultdict(list))   # city → month → [stars]

with open(DATA / "yelp_academic_dataset_review.json", encoding="utf-8") as f:
    for i, line in enumerate(tqdm(f, desc="Reviews")):
        r = json.loads(line)
        bid = r.get("business_id")
        if bid not in biz_ids:
            continue
        city  = city_map[bid]
        month = r["date"][:7]   # "YYYY-MM"
        review_counts[city][month] += 1
        review_stars[city][month].append(r.get("stars", 0))

print("Done streaming reviews.\n")

# ── 3. Stream checkin file → monthly checkin volume per city ─────────────
print("Streaming checkin data...")
checkin_counts = defaultdict(lambda: defaultdict(int))  # city → month → count

with open(DATA / "yelp_academic_dataset_checkin.json", encoding="utf-8") as f:
    for line in tqdm(f, desc="Checkins"):
        c = json.loads(line)
        bid = c.get("business_id")
        if bid not in biz_ids:
            continue
        city = city_map[bid]
        for ts in (c.get("date") or "").split(","):
            ts = ts.strip()
            if len(ts) >= 7:
                checkin_counts[city][ts[:7]] += 1

print("Done streaming checkins.\n")

# ── 4. Build price tier mix per city per month ───────────────────────────
# Use business open/close as proxy: track price tier distribution
# (static snapshot — we'll use review date to approximate opening period)
print("Building price tier signals...")

# For each city, compute static price tier breakdown
price_tier_by_city = {}
for city in CITIES:
    sub = biz_df[biz_df["city"] == city]["price_tier"].dropna()
    total = len(sub)
    if total > 0:
        price_tier_by_city[city] = {
            "pct_budget":   (sub == 1).sum() / total,
            "pct_mid":      (sub == 2).sum() / total,
            "pct_upscale":  (sub == 3).sum() / total,
            "pct_luxury":   (sub == 4).sum() / total,
            "avg_price_tier": sub.mean(),
        }

# ── 5. Assemble monthly signals DataFrame ────────────────────────────────
print("Assembling signals...")
rows = []
for city in CITIES:
    # Collect all months present in review data for this city
    months = sorted(review_counts[city].keys())
    for month in months:
        year = int(month[:4])
        if year < 2005 or year > 2022:
            continue
        n_reviews  = review_counts[city][month]
        stars_list = review_stars[city][month]
        n_checkins = checkin_counts[city].get(month, 0)
        avg_stars  = np.mean(stars_list) if stars_list else np.nan

        # Active restaurant count (stable proxy)
        n_restaurants = len(biz_df[biz_df["city"] == city])
        pt = price_tier_by_city.get(city, {})

        rows.append({
            "city":           city,
            "zillow_metro":   CITIES[city],
            "month":          month,
            "review_volume":  n_reviews,
            "checkin_volume": n_checkins,
            "avg_stars":      avg_stars,
            "n_restaurants":  n_restaurants,
            "reviews_per_restaurant": n_reviews / n_restaurants if n_restaurants else 0,
            **pt,
        })

signals = pd.DataFrame(rows)
signals["month"] = pd.to_datetime(signals["month"])
signals = signals.sort_values(["city", "month"]).reset_index(drop=True)

# ── 6. Add rolling growth rates ──────────────────────────────────────────
for city in CITIES:
    mask = signals["city"] == city
    signals.loc[mask, "review_volume_3m_avg"] = (
        signals.loc[mask, "review_volume"].rolling(3).mean()
    )
    signals.loc[mask, "review_yoy_growth"] = (
        signals.loc[mask, "review_volume"]
        .pct_change(12)
        .mul(100)
    )
    signals.loc[mask, "checkin_yoy_growth"] = (
        signals.loc[mask, "checkin_volume"]
        .pct_change(12)
        .mul(100)
    )

print(f"\nSignals shape: {signals.shape}")
print(signals.groupby("city")[["review_volume","checkin_volume"]].describe().to_string())

# ── 7. Save ───────────────────────────────────────────────────────────────
out_path = OUT / "restaurant_signals.csv"
signals.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
