// Publish step for the Vercel project `zeroth`.
//
// The app is a single self-contained file (`site/index.html`, produced by
// `npm run site` earlier in the repo). This deployment fetches that exact file
// from the pinned commit and publishes it as-is — no install, no bundling, no
// runtime dependency on any CDN.
//
// Why fetch instead of using the source tree? The Vercel project is not linked
// to GitHub (that link needs a "Login Connection" on the Vercel account), so
// deployments here are file uploads. See docs/DEPLOY.md.
import { mkdirSync, writeFileSync } from "node:fs";

const REF = process.env.ZEROTH_REF ?? "59127291dce0f163fbe8e42e18f90a6e9e84f88e";
const URL = `https://raw.githubusercontent.com/yuvii-12345/first/${REF}/site/index.html`;

const res = await fetch(URL);
if (!res.ok) throw new Error(`could not fetch the app: ${res.status} ${res.statusText}`);
const html = await res.text();
if (!html.includes("zeroth") || html.length < 100_000) {
  throw new Error(`unexpected payload (${html.length} bytes)`);
}

mkdirSync("dist", { recursive: true });
writeFileSync("dist/index.html", html);
console.log(`wrote dist/index.html — ${(html.length / 1024).toFixed(1)} kB, self-contained`);
