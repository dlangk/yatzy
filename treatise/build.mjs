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
        out.push(`\n</div>`);
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
        out.push(`<div class="${cls}">\n`);
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
  // Warn and strip em dashes outside <code>/<pre> blocks
  {
    const EM_DASH = /\u2014|&mdash;/g;
    // Split HTML on code/pre tags, only replace in non-code segments
    const parts = html.split(/(<(?:code|pre)[^>]*>[\s\S]*?<\/(?:code|pre)>)/gi);
    let warnings = 0;
    for (let i = 0; i < parts.length; i++) {
      if (i % 2 === 0) { // outside code blocks
        const count = (parts[i].match(EM_DASH) || []).length;
        warnings += count;
        parts[i] = parts[i].replace(EM_DASH, ", ");
      }
    }
    if (warnings > 0) {
      console.warn(`\u26a0  ${file}: found ${warnings} em dash(es) in prose, replaced with ", "`);
    }
    html = parts.join("");
  }
  // Wrap Unicode dice faces (⚀-⚅) in a span for larger rendering
  html = html.replace(/[\u2680-\u2685]/g, ch => `<span class="die-char">${ch}</span>`);
  // Inject section number + permalink into first h2
  const sec = fileSectionMap.get(file);
  if (sec) {
    html = html.replace(
      /<h2>(.+?)<\/h2>/,
      `<h2><a href="#${sec.id}" class="section-link">${sec.number}. $1</a></h2>`
    );
  }
  // Wrap all code blocks with line numbers, hover highlighting, and header.
  // Language-tagged blocks: extract first comment line as header label.
  // Unlabeled blocks: header says "pseudocode".
  html = html.replace(
    /<pre><code(?:\s+class="language-(\w+)")?>([^]*?)<\/code><\/pre>/g,
    (_, lang, body) => {
      const lines = body.replace(/\n$/, "").split("\n");
      let headerText = lang || "pseudocode";

      // For language-tagged blocks, check if first line is a comment with a filename
      if (lang && lines.length > 0) {
        // Match comment patterns: // file.rs — desc, # file.py — desc, -- file.sql
        const commentRe = /^(?:\/\/|#|--)\s*(.+)/;
        const m = lines[0].match(commentRe);
        if (m) {
          headerText = m[1].trim();
          lines.shift(); // remove comment line from body
          // Strip leading blank line after removed comment
          if (lines.length > 0 && lines[0].trim() === "") lines.shift();
        }
      }

      const numbered = lines
        .map((line, i) => `<span class="code-line"><span class="code-ln">${i + 1}</span>${line}</span>`)
        .join("");
      return `<div class="code-card"><div class="code-card-header">${headerText}</div><pre><code>${numbered}</code></pre></div>`;
    }
  );
  // Transform inline [N] markers into superscript ref-marker links.
  // Matches [N] where N is 1-2 digits, but NOT inside <a> tags, code blocks,
  // or Markdown link syntax like [text](url).
  html = html.replace(
    /(<(?:a|code|pre)[^>]*>[\s\S]*?<\/(?:a|code|pre)>)|(\[(\d{1,2})\])/gi,
    (match, tag, bracket, num) => {
      if (tag) return tag; // inside <a>, <code>, or <pre> — leave alone
      return `<a href="#ref-${num}" class="ref-marker" data-ref="${num}">${num}</a>`;
    }
  );
  // Add id="ref-N" anchors to ordered list items inside a .references-list div.
  // The div is emitted by the :::html block in markdown.
  html = html.replace(
    /(<ol[^>]*class="references-list"[^>]*>)([\s\S]*?)(<\/ol>)/g,
    (match, open, body, close) => {
      let i = 0;
      const anchored = body.replace(/<li>/g, () => {
        i++;
        return `<li><a id="ref-${i}"></a>`;
      });
      return open + anchored + close;
    }
  );
  // Inject TOC where placeholder exists
  html = html.replace("<!-- TOC -->", tocHtml);
  const outName = file.replace(/\.md$/, ".html");
  writeFileSync(join(SECTIONS_DIR, outName), html);
  console.log(`  ${file} → ${outName}`);
}

console.log(`Built ${files.length} sections`);
