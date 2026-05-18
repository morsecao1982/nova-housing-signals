"use client";

const SIGNAL_STYLE: Record<string, { bg: string; text: string; border: string; label: string }> = {
  "STRONG BUY":  { bg: "bg-green-900/40",  text: "text-green-300",  border: "border-green-500",  label: "🟢 STRONG BUY"  },
  "BUY":         { bg: "bg-green-900/20",  text: "text-green-400",  border: "border-green-700",  label: "🟩 BUY"         },
  "HOLD":        { bg: "bg-gray-800",      text: "text-gray-300",   border: "border-gray-600",   label: "⬜ HOLD"        },
  "SELL":        { bg: "bg-red-900/20",    text: "text-red-400",    border: "border-red-700",    label: "🟥 SELL"        },
  "STRONG SELL": { bg: "bg-red-900/40",    text: "text-red-300",    border: "border-red-500",    label: "🔴 STRONG SELL" },
};

const PRICE_LABEL: Record<number, string> = { 1: "$", 2: "$$", 3: "$$$", 4: "$$$$" };
const HORIZONS = [
  { key: "1m",  label: "1M"  },
  { key: "3m",  label: "3M"  },
  { key: "6m",  label: "6M"  },
  { key: "12m", label: "12M" },
];

function SignalBadge({ signal }: { signal: string }) {
  const s = SIGNAL_STYLE[signal] || SIGNAL_STYLE["HOLD"];
  return (
    <span className={`px-2 py-0.5 rounded-md text-xs font-bold border ${s.bg} ${s.text} ${s.border}`}>
      {s.label}
    </span>
  );
}

function ScoreBar({ score }: { score: number }) {
  return (
    <div className="flex gap-0.5 items-center">
      {[-2,-1,0,1,2].map(v => (
        <div key={v} className={`h-3 w-4 rounded-sm ${
          v === score
            ? score > 0 ? "bg-green-400" : score < 0 ? "bg-red-400" : "bg-gray-400"
            : "bg-gray-700"
        }`} />
      ))}
    </div>
  );
}

function SummaryBar({ areas }: { areas: [string, any][] }) {
  const counts: Record<string, number> = { "STRONG BUY": 0, "BUY": 0, "HOLD": 0, "SELL": 0, "STRONG SELL": 0 };
  areas.forEach(([, d]) => { if (d.primary_signal in counts) counts[d.primary_signal]++; });
  return (
    <div className="grid grid-cols-5 gap-2 mb-6">
      {(["STRONG BUY","BUY","HOLD","SELL","STRONG SELL"] as const).map(sig => {
        const s = SIGNAL_STYLE[sig];
        return (
          <div key={sig} className={`rounded-xl border p-3 text-center ${s.bg} ${s.border}`}>
            <div className={`text-xl font-black ${s.text}`}>{counts[sig]}</div>
            <div className="text-xs text-gray-400 mt-0.5 leading-tight">{sig}</div>
          </div>
        );
      })}
    </div>
  );
}

function AreaCard({ area, data }: { area: string; data: any }) {
  const s = SIGNAL_STYLE[data.primary_signal] || SIGNAL_STYLE["HOLD"];
  return (
    <div className={`rounded-2xl border p-4 ${s.bg} ${s.border}`}>
      <div className="flex items-start justify-between mb-2">
        <div>
          <h3 className="font-bold">{area}</h3>
          <div className="text-xs text-gray-400">
            {data.n_restaurants} restaurants
            {data.avg_price_tier ? ` · ${PRICE_LABEL[Math.round(data.avg_price_tier)] || ""}` : ""}
            {data.avg_stars ? ` · ⭐ ${data.avg_stars.toFixed(1)}` : ""}
          </div>
        </div>
        <SignalBadge signal={data.primary_signal} />
      </div>

      <ScoreBar score={data.primary_score} />

      <div className="mt-3 grid grid-cols-4 gap-1 text-center text-xs">
        {HORIZONS.map(({ key, label }) => {
          const h = data.horizons?.[key];
          if (!h) return null;
          const hs = SIGNAL_STYLE[h.signal] || SIGNAL_STYLE["HOLD"];
          return (
            <div key={key} className="bg-gray-900/50 rounded-lg p-1.5">
              <div className="text-gray-500 text-[10px]">{label}</div>
              <div className={`font-bold text-xs mt-0.5 ${hs.text}`}>
                {h.signal === "STRONG BUY" ? "S.BUY" : h.signal === "STRONG SELL" ? "S.SELL" : h.signal}
              </div>
              <div className="text-gray-400 text-[10px]">
                {h.pred_pct != null ? `${h.pred_pct > 0 ? "+" : ""}${h.pred_pct.toFixed(1)}%` : "—"}
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-2 flex gap-3 text-xs text-gray-400">
        <span>MoM: <span className={data.review_mom > 0 ? "text-green-400" : data.review_mom < 0 ? "text-red-400" : "text-gray-300"}>
          {data.review_mom != null ? `${data.review_mom > 0 ? "+" : ""}${data.review_mom.toFixed(1)}%` : "—"}
        </span></span>
        <span>YoY: <span className={data.review_yoy > 0 ? "text-green-400" : data.review_yoy < 0 ? "text-red-400" : "text-gray-300"}>
          {data.review_yoy != null ? `${data.review_yoy > 0 ? "+" : ""}${data.review_yoy.toFixed(1)}%` : "—"}
        </span></span>
      </div>
    </div>
  );
}

export default function NovaClient({ results }: { results: any }) {
  if (!results) {
    return (
      <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center gap-4">
        <div className="text-6xl">🏗️</div>
        <h1 className="text-2xl font-black text-yellow-400">Data Collection In Progress</h1>
        <p className="text-gray-400 max-w-md text-center">
          The first monthly snapshot hasn&apos;t been collected yet.
          The pipeline runs on the 1st of every month.
        </p>
      </main>
    );
  }

  const allAreas = Object.entries(results.areas as Record<string, any>)
    .sort((a, b) => b[1].primary_score - a[1].primary_score);

  // Group by region
  const regions: Record<string, [string, any][]> = {};
  allAreas.forEach(([area, data]) => {
    const r = data.region || "Northern Virginia";
    if (!regions[r]) regions[r] = [];
    regions[r].push([area, data]);
  });

  const regionOrder = ["Northern Virginia", "Montgomery County MD"];

  return (
    <main className="min-h-screen bg-gray-950 text-white p-6">
      <div className="max-w-6xl mx-auto">

        {/* Header */}
        <div className="mb-8">
          <div className="flex items-start justify-between flex-wrap gap-4">
            <div>
              <h1 className="text-3xl font-black text-yellow-400">DMV Housing Signals</h1>
              <p className="text-gray-400 mt-1">Northern Virginia · Montgomery County MD</p>
              <p className="text-gray-500 text-sm mt-0.5">Based on monthly restaurant activity data</p>
            </div>
            <div className="text-right text-sm text-gray-500">
              <div>Data: <span className="text-gray-300">{results.data_month}</span></div>
              <div>Confidence: <span className={
                results.confidence === "high" ? "text-green-400" :
                results.confidence === "medium" ? "text-yellow-400" : "text-red-400"
              }>{results.confidence}</span></div>
              {results.bootstrap && (
                <div className="text-yellow-600 text-xs mt-1">⚠ Bootstrap — &lt;12 months data</div>
              )}
            </div>
          </div>
        </div>

        {/* Overall summary */}
        <SummaryBar areas={allAreas} />

        {/* Regions */}
        {regionOrder.map(region => {
          const areas = regions[region];
          if (!areas || areas.length === 0) return null;
          return (
            <div key={region} className="mb-10">
              <div className="flex items-center gap-3 mb-4">
                <h2 className="text-xl font-black text-white">{region}</h2>
                <div className="h-px flex-1 bg-gray-700" />
                <span className="text-gray-500 text-sm">{areas.length} areas</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {areas.map(([area, data]) => (
                  <AreaCard key={area} area={area} data={data} />
                ))}
              </div>
            </div>
          );
        })}

        {/* Methodology */}
        <div className="mt-6 border border-gray-800 rounded-2xl p-5 text-sm text-gray-500">
          <h3 className="text-gray-300 font-bold mb-2">How This Works</h3>
          <p className="mb-2">
            Restaurant activity (review volume, foot traffic trends, price tier mix) is collected monthly
            from Google Places API. Two ML models — a GBM regression (predicts % price change) and a
            classifier (predicts direction) — generate combined signals at 1, 3, 6, and 12-month horizons.
            Primary signal uses the 6-month horizon (best accuracy).
          </p>
          <p className="text-gray-600 text-xs">
            ⚠️ Research tool only, not financial advice. Model trained on 40 US cities — DC metro
            predictions carry additional uncertainty. Always combine with local market knowledge.
          </p>
        </div>
      </div>
    </main>
  );
}
