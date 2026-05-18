"""
Monthly Google Places data collector for Northern Virginia.
Snapshots all restaurants per area, saves to data/nova_snapshots/YYYY-MM.json

Run: python nova/collect_data.py
"""
import os, json, time, sys
from pathlib import Path
from datetime import datetime
import googlemaps

API_KEY   = os.environ.get("GOOGLE_PLACES_API_KEY", "")
OUT_DIR   = Path(__file__).parent.parent / "data" / "nova_snapshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Northern Virginia search areas: (name, lat, lng, radius_m)
NOVA_AREAS = [
    ("Arlington",       38.8816, -77.0910, 5000),
    ("Alexandria",      38.8048, -77.0469, 5000),
    ("Falls Church",    38.8823, -77.1711, 4000),
    ("McLean",          38.9339, -77.1773, 4000),
    ("Fairfax",         38.8462, -77.3064, 5000),
    ("Vienna",          38.9012, -77.2653, 4000),
    ("Reston",          38.9587, -77.3570, 5000),
    ("Herndon",         38.9696, -77.3861, 4000),
    ("Chantilly",       38.8851, -77.4319, 5000),
    ("Ashburn",         39.0437, -77.4875, 5000),
    ("Sterling",        39.0026, -77.4294, 4000),
    ("Leesburg",        39.1157, -77.5636, 5000),
    ("Woodbridge",      38.6543, -77.2497, 5000),
    ("Manassas",        38.7509, -77.4753, 5000),
]

RESTAURANT_TYPES = ["restaurant", "cafe", "bar", "bakery", "meal_takeaway"]

def fetch_area_restaurants(gmaps, area_name, lat, lng, radius):
    """Fetch all restaurants for one area using nearby search + pagination."""
    restaurants = {}
    for rtype in RESTAURANT_TYPES[:2]:   # restaurant + cafe to stay in budget
        try:
            resp = gmaps.places_nearby(
                location=(lat, lng), radius=radius, type=rtype
            )
            results = resp.get("results", [])
            # Paginate up to 2 more pages (60 results total per type)
            for _ in range(2):
                token = resp.get("next_page_token")
                if not token: break
                time.sleep(2)   # Required delay before next_page_token is valid
                resp = gmaps.places_nearby(page_token=token)
                results.extend(resp.get("results", []))

            for r in results:
                pid = r.get("place_id", "")
                if not pid or pid in restaurants: continue
                restaurants[pid] = {
                    "place_id":    pid,
                    "name":        r.get("name", ""),
                    "rating":      r.get("rating"),
                    "review_count":r.get("user_ratings_total", 0),
                    "price_level": r.get("price_level"),
                    "lat":         r["geometry"]["location"]["lat"],
                    "lng":         r["geometry"]["location"]["lng"],
                    "area":        area_name,
                    "types":       r.get("types", []),
                }
        except Exception as e:
            print(f"    Warning: {area_name}/{rtype}: {e}")

    return list(restaurants.values())

def run():
    if not API_KEY:
        print("ERROR: GOOGLE_PLACES_API_KEY not set")
        sys.exit(1)

    gmaps    = googlemaps.Client(key=API_KEY)
    month    = datetime.now().strftime("%Y-%m")
    out_file = OUT_DIR / f"{month}.json"

    if out_file.exists():
        print(f"Snapshot {month} already exists — skipping collection.")
        return

    print(f"Collecting NoVA restaurant data for {month}...")
    snapshot = {"month": month, "collected_at": datetime.now().isoformat(), "areas": {}}

    total = 0
    for area_name, lat, lng, radius in NOVA_AREAS:
        print(f"  {area_name}...", end=" ", flush=True)
        rests = fetch_area_restaurants(gmaps, area_name, lat, lng, radius)
        snapshot["areas"][area_name] = rests
        total += len(rests)
        print(f"{len(rests)} restaurants")
        time.sleep(0.5)

    with open(out_file, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"\nSaved {total} restaurants → {out_file}")

if __name__ == "__main__":
    run()
