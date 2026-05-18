import NovaClient from "./NovaClient";

// Direct import — bundled at build time.
// When GitHub Actions updates nova_results.json and pushes,
// Vercel rebuilds automatically and picks up fresh data.
let results: any = null;
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  results = require("../../public/nova_results.json");
} catch {
  results = null;
}

export default function NovaPage() {
  return <NovaClient results={results} />;
}
