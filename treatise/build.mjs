import { readFileSync, writeFileSync, readdirSync } from "fs";
import { join, basename } from "path";
import MarkdownIt from "markdown-it";

const SECTIONS_DIR = join(import.meta.dirname, "sections");
const md = new MarkdownIt({ html: true });

// Block shortcodes: :::type{#id} ... :::
// Supports: section, math, code, equation, insight, part-title, html
const BLOCK_RE = /^:::(\w[\w-]*)(\{#([\w-]+)\})?\s*$/;

// Map shortcode names to CSS classes (identity if not listed)
const CLASS_MAP = { code: "code-detail" };

function processShortcodes(src) {
  const lines = src.split("\n");
  const out = [];
  const stack = []; // track nested blocks
  let htmlBlock = null; // accumulator for :::html blocks

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // Closing fence
    if (trimmed === ":::" && stack.length > 0) {
      const top = stack.pop();
      if (top.type === "html") {
        // End html passthrough — emit accumulated content raw
        out.push(htmlBlock.join("\n"));
        htmlBlock = null;
      } else if (top.type === "section") {
        out.push("</section>");
      } else if (top.type === "part-title") {
        out.push("</div>");
      } else {
        // math, code, equation, insight
        out.push(`</div>`);
      }
      continue;
    }

    // Opening fence
    const m = trimmed.match(BLOCK_RE);
    if (m) {
      const type = m[1];
      const id = m[3] || null;

      if (type === "html") {
        stack.push({ type: "html" });
        htmlBlock = [];
      } else if (type === "section") {
        stack.push({ type: "section" });
        out.push(id ? `<section id="${id}">` : "<section>");
      } else if (type === "part-title") {
        stack.push({ type: "part-title" });
        out.push(`<div class="part-title">`);
      } else {
        // math, code, equation, insight
        const cls = CLASS_MAP[type] || type;
        stack.push({ type });
        out.push(`<div class="${cls}">`);
      }
      continue;
    }

    // Inside html block — accumulate raw
    if (htmlBlock !== null) {
      htmlBlock.push(line);
      continue;
    }

    // Inline shortcode: ::concept[display text]{slug}
    const processed = line.replace(
      /::concept\[([^\]]+)\]\{([^}]+)\}/g,
      '<span class="concept" data-concept="$2">$1</span>'
    );
    out.push(processed);
  }

  return out.join("\n");
}

// Process all .md files in sections/
const files = readdirSync(SECTIONS_DIR)
  .filter((f) => f.endsWith(".md"))
  .sort();

if (files.length === 0) {
  console.error("No .md files found in", SECTIONS_DIR);
  process.exit(1);
}

// Build TOC from numbered section files (01-*.md through 99-*.md)
const SECTION_ID_RE = /^:::section\{#([\w-]+)\}/m;
const FIRST_H2_RE = /^## (.+)$/m;
const PART_LABEL_RE = /^:::part-title\s*\n([^\n]+)\n:::/m;

const tocEntries = [];
const fileSectionMap = new Map(); // file -> { number, id }
let sectionNum = 0;
for (const file of files) {
  if (!/^\d/.test(file) || file === "header.md" || file.startsWith("00-")) continue;
  const src = readFileSync(join(SECTIONS_DIR, file), "utf-8");
  const idMatch = src.match(SECTION_ID_RE);
  const h2Match = src.match(FIRST_H2_RE);
  if (!idMatch || !h2Match) continue;
  sectionNum++;
  const id = idMatch[1];
  const title = h2Match[1].replace(/^\d+\.\s*/, ""); // strip leading "1. "
  tocEntries.push({ id, display: title, number: sectionNum });
  fileSectionMap.set(file, { number: sectionNum, id });
}

const tocHtml =
  `<nav class="toc">\n    <h3>Contents</h3>\n    <ol>\n` +
  tocEntries.map((e) => `      <li><a href="#${e.id}">${e.display}</a></li>`).join("\n") +
  `\n    </ol>\n  </nav>`;

// Build all sections
for (const file of files) {
  const src = readFileSync(join(SECTIONS_DIR, file), "utf-8");
  const shortcoded = processShortcodes(src);
  let html = md.render(shortcoded);
  // Kill all em dashes (Unicode and HTML entity)
  html = html.replace(/\u2014/g, "--").replace(/&mdash;/g, "--");
  // Inject section number + permalink into first h2
  const sec = fileSectionMap.get(file);
  if (sec) {
    html = html.replace(
      /<h2>(.+?)<\/h2>/,
      `<h2><a href="#${sec.id}" class="section-link">${sec.number}. $1</a></h2>`
    );
  }
  // Inject TOC where placeholder exists
  html = html.replace("<!-- TOC -->", tocHtml);
  const outName = file.replace(/\.md$/, ".html");
  writeFileSync(join(SECTIONS_DIR, outName), html);
  console.log(`  ${file} → ${outName}`);
}

console.log(`Built ${files.length} sections`);
