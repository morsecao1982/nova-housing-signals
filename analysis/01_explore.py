"""
Quick data exploration — understand what's in the Yelp dataset
before building the full analysis pipeline.
"""
import json
import pandas as pd
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent.parent / "data" / "Yelp JSON"

# ── 1. Load business data ──────────────────────────────────────────────────
print("Loading business data...")
businesses = []
with open(DATA / "yelp_academic_dataset_business.json", encoding="utf-8") as f:
    for line in f:
        businesses.append(json.loads(line))

df = pd.DataFrame(businesses)
print(f"Total businesses: {len(df):,}")
print(f"Columns: {list(df.columns)}\n")

# ── 2. Filter to restaurants ───────────────────────────────────────────────
def is_restaurant(cats):
    if not cats or not isinstance(cats, str):
        return False
    return any(c.strip() in {
        "Restaurants", "Food", "Bars", "Nightlife",
        "Fast Food", "Pizza", "Burgers", "Sandwiches",
        "Mexican", "Chinese", "Italian", "American (New)",
        "American (Traditional)", "Cafes", "Breakfast & Brunch",
    } for c in cats.split(","))

df["is_restaurant"] = df["categories"].apply(is_restaurant)
restaurants = df[df["is_restaurant"]].copy()
print(f"Restaurants: {len(restaurants):,} ({len(restaurants)/len(df)*100:.1f}% of all businesses)\n")

# ── 3. Metro breakdown ─────────────────────────────────────────────────────
print("── Top metros (by restaurant count) ──")
metro = restaurants.groupby("city").agg(
    count=("business_id", "count"),
    avg_stars=("stars", "mean"),
    avg_reviews=("review_count", "mean"),
).sort_values("count", ascending=False).head(20)
print(metro.to_string())
print()

# ── 4. State breakdown ─────────────────────────────────────────────────────
print("── By state ──")
print(restaurants["state"].value_counts().head(15).to_string())
print()

# ── 5. Price tier distribution ────────────────────────────────────────────
print("── Price tier distribution ──")
price = restaurants["attributes"].apply(
    lambda x: x.get("RestaurantsPriceRange2") if isinstance(x, dict) else None
)
print(price.value_counts().to_string())
print()

# ── 6. Open vs closed ─────────────────────────────────────────────────────
print("── Open vs closed ──")
print(restaurants["is_open"].value_counts().rename({1: "Open", 0: "Closed"}).to_string())
print()

# ── 7. Sample a few review dates ──────────────────────────────────────────
print("── Sampling review data for date range ──")
sample_reviews = []
with open(DATA / "yelp_academic_dataset_review.json", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 10000:
            break
        r = json.loads(line)
        sample_reviews.append(r["date"])

dates = pd.Series(pd.to_datetime(sample_reviews))
print(f"Review date range (first 10k): {dates.min().date()} → {dates.max().date()}")
print(f"Year distribution:\n{dates.dt.year.value_counts().sort_index().to_string()}")
print()

# ── 8. Checkin sample ─────────────────────────────────────────────────────
print("── Checkin data sample ──")
with open(DATA / "yelp_academic_dataset_checkin.json", encoding="utf-8") as f:
    sample = json.loads(f.readline())
print(f"Sample checkin keys: {list(sample.keys())}")
print(f"Sample dates field (first 200 chars): {str(sample.get('date',''))[:200]}")
