import { readFileSync, writeFileSync, readdirSync } from "fs";
import { join, basename } from "path";
import MarkdownIt from "markdown-it";

const SECTIONS_DIR = join(import.meta.dirname, "sections");
const md = new MarkdownIt({ html: true });

// Block shortcodes: :::type{#id} ... :::
// Supports: section, depth-2, depth-3, equation, insight, part-title, html
const BLOCK_RE = /^:::(\w[\w-]*)(\{#([\w-]+)\})?\s*$/;

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
        // depth-2, depth-3, equation, insight
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
        // depth-2, depth-3, equation, insight
        stack.push({ type });
        out.push(`<div class="${type}">`);
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
const files = readdirSync(SECTIONS_DIR).filter((f) => f.endsWith(".md"));

if (files.length === 0) {
  console.error("No .md files found in", SECTIONS_DIR);
  process.exit(1);
}

for (const file of files) {
  const src = readFileSync(join(SECTIONS_DIR, file), "utf-8");
  const shortcoded = processShortcodes(src);
  let html = md.render(shortcoded);
  // Kill all em dashes (Unicode and HTML entity)
  html = html.replace(/\u2014/g, "--").replace(/&mdash;/g, "--");
  const outName = file.replace(/\.md$/, ".html");
  writeFileSync(join(SECTIONS_DIR, outName), html);
  console.log(`  ${file} → ${outName}`);
}

console.log(`Built ${files.length} sections`);
