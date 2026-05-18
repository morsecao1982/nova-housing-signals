"use client";

const SIGNAL_STYLE: Record<string, { bg: string; text: string; border: string; label: string }> = {
  "STRONG BUY":  { bg: "bg-green-900/40",  text: "text-green-300",  border: "border-green-500",  label: "🟢 STRONG BUY"  },
  "BUY":         { bg: "bg-green-900/20",  text: "text-green-400",  border: "border-green-700",  label: "🟩 BUY"         },
  "HOLD":        { bg: "bg-gray-800",      text: "text-gray-300",   border: "border-gray-600",   label: "⬜ HOLD"        },
  "SELL":        { bg: "bg-red-900/20",    text: "text-red-400",    border: "border-red-700",    label: "🟥 SELL"        },
  "STRONG SELL": { bg: "bg-red-900/40",    text: "text-red-300",    border: "border-red-500",    label: "🔴 STRONG SELL" },
};

const PRICE_LABEL = ["", "$", "$$", "$$$", "$$$$"];
const HORIZONS    = [
  { key: "1m",  label: "1 Month"  },
  { key: "3m",  label: "3 Months" },
  { key: "6m",  label: "6 Months" },
  { key: "12m", label: "12 Months"},
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
  const colors = [-2,2].map(_ => "bg-gray-700");
  const vals   = [-2,-1,0,1,2];
  return (
    <div className="flex gap-0.5 items-center">
      {vals.map(v => (
        <div key={v} className={`h-3 w-4 rounded-sm ${
          v === score
            ? score > 0 ? "bg-green-400" : score < 0 ? "bg-red-400" : "bg-gray-400"
            : "bg-gray-700"
        }`} />
      ))}
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
          The first monthly snapshot hasn't been collected yet.
          The pipeline runs on the 1st of every month.
          Check back soon.
        </p>
      </main>
    );
  }

  const areas      = Object.entries(results.areas as Record<string, any>)
    .sort((a, b) => b[1].primary_score - a[1].primary_score);
  const dataDate   = results.data_month;
  const confidence = results.confidence;
  const bootstrap  = results.bootstrap;

  const counts = { "STRONG BUY": 0, "BUY": 0, "HOLD": 0, "SELL": 0, "STRONG SELL": 0 };
  areas.forEach(([, d]) => { if (d.primary_signal in counts) counts[d.primary_signal as keyof typeof counts]++; });

  return (
    <main className="min-h-screen bg-gray-950 text-white p-6">
      <div className="max-w-6xl mx-auto">

        {/* Header */}
        <div className="mb-8">
          <a href="/" className="text-gray-500 hover:text-white text-sm">← Back</a>
          <div className="mt-4 flex items-start justify-between flex-wrap gap-4">
            <div>
              <h1 className="text-3xl font-black text-yellow-400">Northern Virginia</h1>
              <p className="text-gray-400 mt-1">Housing Market Signals · Based on Restaurant Activity</p>
            </div>
            <div className="text-right text-sm text-gray-500">
              <div>Data: <span className="text-gray-300">{dataDate}</span></div>
              <div>Confidence: <span className={confidence === "high" ? "text-green-400" : confidence === "medium" ? "text-yellow-400" : "text-red-400"}>{confidence}</span></div>
              {bootstrap && <div className="text-yellow-600 text-xs mt-1">⚠ Bootstrap mode — &lt;12 months data</div>}
            </div>
          </div>
        </div>

        {/* Summary bar */}
        <div className="grid grid-cols-5 gap-3 mb-8">
          {(["STRONG BUY","BUY","HOLD","SELL","STRONG SELL"] as const).map(sig => {
            const s = SIGNAL_STYLE[sig];
            return (
              <div key={sig} className={`rounded-xl border p-4 text-center ${s.bg} ${s.border}`}>
                <div className={`text-2xl font-black ${s.text}`}>{counts[sig]}</div>
                <div className="text-xs text-gray-400 mt-1">{sig}</div>
              </div>
            );
          })}
        </div>

        {/* Area cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {areas.map(([area, data]) => {
            const s = SIGNAL_STYLE[data.primary_signal] || SIGNAL_STYLE["HOLD"];
            return (
              <div key={area} className={`rounded-2xl border p-5 ${s.bg} ${s.border}`}>
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h2 className="text-lg font-bold">{area}</h2>
                    <div className="text-xs text-gray-400">
                      {data.n_restaurants} restaurants · {PRICE_LABEL[Math.round(data.avg_price_tier)] || "—"} · ⭐ {data.avg_stars?.toFixed(1) || "—"}
                    </div>
                  </div>
                  <SignalBadge signal={data.primary_signal} />
                </div>

                <ScoreBar score={data.primary_score} />

                {/* Horizon table */}
                <div className="mt-4 grid grid-cols-4 gap-1 text-center text-xs">
                  {HORIZONS.map(({ key, label }) => {
                    const h = data.horizons?.[key];
                    if (!h) return null;
                    const hs = SIGNAL_STYLE[h.signal] || SIGNAL_STYLE["HOLD"];
                    return (
                      <div key={key} className="bg-gray-900/50 rounded-lg p-2">
                        <div className="text-gray-500 text-[10px]">{label}</div>
                        <div className={`font-bold text-sm mt-0.5 ${hs.text}`}>{h.signal.replace(" BUY","🟢").replace(" SELL","🔴").replace("HOLD","⬜").replace("STRONG 🟢","🟢★").replace("STRONG 🔴","🔴★")}</div>
                        <div className="text-gray-400 text-[10px] mt-0.5">{h.pred_pct > 0 ? "+" : ""}{h.pred_pct?.toFixed(1)}%</div>
                      </div>
                    );
                  })}
                </div>

                {/* Restaurant signal */}
                <div className="mt-3 flex gap-4 text-xs text-gray-400">
                  <span>Review MoM: <span className={data.review_mom > 0 ? "text-green-400" : data.review_mom < 0 ? "text-red-400" : "text-gray-300"}>
                    {data.review_mom > 0 ? "+" : ""}{data.review_mom?.toFixed(1)}%
                  </span></span>
                  <span>YoY: <span className={data.review_yoy > 0 ? "text-green-400" : data.review_yoy < 0 ? "text-red-400" : "text-gray-300"}>
                    {data.review_yoy > 0 ? "+" : ""}{data.review_yoy?.toFixed(1)}%
                  </span></span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Methodology note */}
        <div className="mt-10 border border-gray-800 rounded-2xl p-6 text-sm text-gray-500">
          <h3 className="text-gray-300 font-bold mb-2">How This Works</h3>
          <p className="mb-2">
            Restaurant activity (review volume, foot traffic trends, price tier mix) is collected monthly
            from Google Places API across 14 Northern Virginia areas. Two ML models — a regression model
            (predicts % price change) and a classifier (predicts direction) — generate a combined
            BUY/SELL/HOLD signal for each area at 1, 3, 6, and 12-month horizons.
          </p>
          <p className="mb-2">
            <span className="text-yellow-400 font-bold">Primary signal: 6-month horizon</span> —
            best balance of accuracy (91%) and lead time. The model catches ~46% of downturns.
          </p>
          <p className="text-gray-600">
            ⚠️ This is a research tool, not financial advice. Model was trained on 40 US cities;
            Northern Virginia predictions carry additional uncertainty due to domain shift.
            Always combine with local market knowledge before making decisions.
          </p>
        </div>
      </div>
    </main>
  );
}
