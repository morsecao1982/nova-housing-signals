import { readFileSync, existsSync } from "fs";
import { join }                     from "path";
import NovaClient                   from "./NovaClient";

interface AreaData {
  primary_signal: string;
  primary_score:  number;
  n_restaurants:  number;
  avg_stars:      number;
  avg_price_tier: number;
  review_mom:     number;
  review_yoy:     number;
  horizons: Record<string, {
    signal: string; score: number;
    pred_pct: number; down_prob: number; up_prob: number;
  }>;
}

interface Results {
  generated_at: string;
  data_month:   string;
  bootstrap:    boolean;
  confidence:   string;
  areas:        Record<string, AreaData>;
}

function loadResults(): Results | null {
  const p = join(process.cwd(), "public", "nova_results.json");
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, "utf-8"));
}

export default function NovaPage() {
  const results = loadResults();
  return <NovaClient results={results} />;
}

export const revalidate = 3600; // revalidate every hour
