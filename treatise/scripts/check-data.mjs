#!/usr/bin/env node
// Verify every data file the treatise charts ACTUALLY NEED exists in
// treatise/data/. Run before deploy so an incomplete working tree can never
// wipe the server via `rsync --delete`.
//
//   node treatise/scripts/check-data.mjs        (from repo root or treatise/)
//   just check-treatise-data
//
// A data file is "required" iff some chart that loads it has a container
// (`id="chart-..."`) present in a section markdown file. Charts whose
// container was never placed in the prose ("phantom" charts) do not create a
// requirement — their modules and DataLoader entries can exist without the
// data. This keeps the required-set tied to what the site actually renders.
//
// Exit 0 = all required files present and non-empty. Exit 1 = something missing.

import { readFileSync, readdirSync, existsSync, statSync } from "fs";
import { join, dirname, resolve, basename } from "path";
import { fileURLToPath } from "url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const treatiseDir = resolve(scriptDir, "..");
const dataDir = join(treatiseDir, "data");
const chartsDir = join(treatiseDir, "js", "charts");
const indexHtml = readFileSync(join(treatiseDir, "index.html"), "utf8");
const dataLoaderSrc = readFileSync(join(treatiseDir, "js", "data-loader.js"), "utf8");

// 1. DataLoader method -> file name.  e.g. kdeCurves: () => load('kde_curves.json')
const methodToFile = {};
for (const m of dataLoaderSrc.matchAll(/(\w+):\s*\(\)\s*=>\s*load\(\s*['"]([^'"]+)['"]/g)) {
  methodToFile[m[1]] = m[2];
}

// 2. index.html registry: container id -> init function name.
const containerToInit = {};
for (const m of indexHtml.matchAll(/'(chart-[a-z0-9-]+)':\s*\{\s*init:\s*(\w+)/g)) {
  containerToInit[m[1]] = m[2];
}
// init function -> chart module basename (from the imports).
const initToModule = {};
for (const m of indexHtml.matchAll(/import\s*\{([^}]+)\}\s*from\s*'\.\/js\/charts\/([^']+)'/g)) {
  const mod = basename(m[2], ".js");
  for (const fn of m[1].split(",").map((s) => s.trim()).filter(Boolean)) initToModule[fn] = mod;
}

// 3. Live container ids = those appearing in any section markdown.
const sectionsDir = join(treatiseDir, "sections");
const liveContainers = new Set();
for (const f of readdirSync(sectionsDir)) {
  if (!f.endsWith(".md")) continue;
  const src = readFileSync(join(sectionsDir, f), "utf8");
  for (const m of src.matchAll(/id=["'](chart-[a-z0-9-]+)["']/g)) liveContainers.add(m[1]);
}

// 4. For every chart MODULE, the set of data files it loads.
//    Supports DataLoader.method() and direct fetch(`data/foo.json`).
function filesLoadedBy(moduleBase) {
  const p = join(chartsDir, moduleBase + ".js");
  if (!existsSync(p)) return [];
  const src = readFileSync(p, "utf8");
  const files = new Set();
  for (const m of src.matchAll(/DataLoader\.(\w+)\s*\(/g)) {
    if (methodToFile[m[1]]) files.add(methodToFile[m[1]]);
  }
  for (const m of src.matchAll(/fetch\(\s*[`'"]data\/([^`'"?]+\.json)/g)) files.add(m[1]);
  return [...files];
}

// 5. Required = union of files loaded by charts whose container is live.
const required = new Set();
const requiredBy = {};
for (const [container, init] of Object.entries(containerToInit)) {
  if (!liveContainers.has(container)) continue; // phantom chart -> no requirement
  const mod = initToModule[init];
  if (!mod) continue;
  for (const f of filesLoadedBy(mod)) {
    required.add(f);
    (requiredBy[f] ||= new Set()).add(container);
  }
}

const req = [...required].sort();
const missing = req.filter((f) => !existsSync(join(dataDir, f)));
const empty = req.filter(
  (f) => existsSync(join(dataDir, f)) && statSync(join(dataDir, f)).size === 0
);

console.log(
  `Checked ${req.length} required data files (loaded by ${liveContainers.size} live chart containers)`
);
if (missing.length === 0 && empty.length === 0) {
  console.log("✓ All present");
  process.exit(0);
}
for (const [label, list] of [["MISSING", missing], ["EMPTY", empty]]) {
  if (!list.length) continue;
  console.error(`\n✗ ${label} (${list.length}):`);
  for (const f of list) console.error(`    ${f}  ← ${[...requiredBy[f]].join(", ")}`);
}
console.error(
  "\nRegenerate with `just regen-treatise-data`, then retry. Aborting to avoid deploying an incomplete data set."
);
process.exit(1);
