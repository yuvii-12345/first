/**
 * Produce a self-contained copy of the built app in `site/`:
 *
 *   site/index.html   the app, with CSS and JS inlined
 *   site/app.js       the whole application (worker inlined as a blob)
 *   site/app.css      the stylesheet
 *
 * The split files exist so the app can also be served from a static host that
 * pulls individual files (jsDelivr, a raw CDN, an S3 bucket) while `index.html`
 * stays a single file you can double-click.
 *
 *   npm run site
 */
import { execSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

const read = (p) => readFileSync(p, "utf8");

execSync("npm run build", { stdio: "inherit" });

const html = read("dist/index.html");
const jsMatch = html.match(/<script type="module"[^>]*src="([^"]+)"/);
const cssMatch = html.match(/<link rel="stylesheet"[^>]*href="([^"]+)"/);
if (!jsMatch) throw new Error("could not find the entry script in dist/index.html");

const jsPath = "dist" + jsMatch[1];
const cssPath = cssMatch ? "dist" + cssMatch[1] : null;
const js = read(jsPath);
const css = cssPath ? read(cssPath) : "";

// Fold the stylesheet into the script so a host has to fetch exactly one file.
const injector =
  `(function(){try{var s=document.createElement("style");s.setAttribute("data-zeroth","1");` +
  `s.textContent=${JSON.stringify(css)};document.head.appendChild(s)}catch(e){}})();\n`;
const bundled = injector + js;

mkdirSync("site", { recursive: true });
writeFileSync("site/app.js", bundled);
writeFileSync("site/app.css", css);

// The standalone page: same markup, everything inlined, no network at all.
const inline = html
  .replace(/<script type="module"[^>]*src="[^"]+"><\/script>/, `<script type="module">\n${bundled}\n</script>`)
  .replace(/<link rel="stylesheet"[^>]*>/, "");
writeFileSync("site/index.html", inline);

const kb = (n) => `${(n / 1024).toFixed(1)} kB`;
console.log(`site/app.js    ${kb(bundled.length)} (styles inlined)`);
console.log(`site/app.css   ${kb(css.length)}`);
console.log(`site/index.html ${kb(inline.length)} (single file)`);
