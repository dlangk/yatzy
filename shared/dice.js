/**
 * Shared die renderer — single source of truth for all SVG pip dice.
 * Used by treatise, play UI, and profiler.
 */

export const PIPS = [
  [],
  [{ cx: 24, cy: 24 }],
  [{ cx: 14, cy: 14 }, { cx: 34, cy: 34 }],
  [{ cx: 14, cy: 14 }, { cx: 24, cy: 24 }, { cx: 34, cy: 34 }],
  [{ cx: 14, cy: 14 }, { cx: 34, cy: 14 }, { cx: 14, cy: 34 }, { cx: 34, cy: 34 }],
  [{ cx: 14, cy: 14 }, { cx: 34, cy: 14 }, { cx: 24, cy: 24 }, { cx: 14, cy: 34 }, { cx: 34, cy: 34 }],
  [{ cx: 14, cy: 12 }, { cx: 34, cy: 12 }, { cx: 14, cy: 24 }, { cx: 34, cy: 24 }, { cx: 14, cy: 36 }, { cx: 34, cy: 36 }],
];

const STATES = {
  normal:         { fill: 'var(--bg-alt)', stroke: 'var(--border)',        sw: 2, dash: '',    pipOp: 1   },
  kept:           { fill: 'var(--bg)',     stroke: 'var(--accent)',         sw: 3, dash: '',    pipOp: 1   },
  reroll:         { fill: 'none',          stroke: 'var(--border)',         sw: 2, dash: '4 3', pipOp: 0.2 },
  'will-reroll':  { fill: 'var(--bg-alt)', stroke: 'var(--border)',        sw: 2, dash: '',    pipOp: 0.45 },
  'optimal-keep': { fill: 'var(--bg)',     stroke: 'var(--color-success)',  sw: 3, dash: '',    pipOp: 1   },
  'optimal-reroll': { fill: 'var(--bg-alt)', stroke: 'var(--color-danger)', sw: 3, dash: '',   pipOp: 0.45 },
  faded:          { fill: 'var(--bg-alt)', stroke: 'var(--border)',         sw: 2, dash: '',    pipOp: 0.5 },
};

function dieMarkupCustom(value, fill, stroke, sw, dash, pipOp) {
  const dashAttr = dash ? ` stroke-dasharray="${dash}"` : '';
  let svg = `<rect x="1" y="1" width="46" height="46" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dashAttr}/>`;
  if (value === 0) {
    svg += `<text x="24" y="24" text-anchor="middle" dy="0.35em" font-size="20" font-weight="600" fill="var(--text)" opacity="0.3">?</text>`;
  } else {
    (PIPS[value] || []).forEach(p => {
      svg += `<circle cx="${p.cx}" cy="${p.cy}" r="4.5" fill="var(--text)" opacity="${pipOp}"/>`;
    });
  }
  return svg;
}

function dieMarkup(value, state) {
  const s = STATES[state] || STATES.normal;
  const dashAttr = s.dash ? ` stroke-dasharray="${s.dash}"` : '';
  let svg = `<rect x="1" y="1" width="46" height="46" rx="8" fill="${s.fill}" stroke="${s.stroke}" stroke-width="${s.sw}"${dashAttr}/>`;
  if (value === 0) {
    // Blank die: show "?" placeholder
    svg += `<text x="24" y="24" text-anchor="middle" dy="0.35em" font-size="20" font-weight="600" fill="var(--text)" opacity="0.3">?</text>`;
  } else {
    (PIPS[value] || []).forEach(p => {
      svg += `<circle cx="${p.cx}" cy="${p.cy}" r="4.5" fill="var(--text)" opacity="${s.pipOp}"/>`;
    });
  }
  return svg;
}

/**
 * Create an SVG DOM element for a die face.
 * @param {number} value - Die value 1-6 (0 for blank)
 * @param {object} options
 * @param {number} options.size - Width/height in px (default 48)
 * @param {string} options.state - Visual state (default 'normal')
 * @param {boolean} options.clickable - Add pointer cursor (default false)
 * @returns {SVGSVGElement}
 */
export function createDieSVG(value, options = {}) {
  const { size = 48, state = 'normal', clickable = false,
          fill, stroke, sw, pipOp } = options;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 48 48');
  svg.setAttribute('width', String(size));
  svg.setAttribute('height', String(size));
  svg.classList.add('die-svg');
  if (clickable) svg.style.cursor = 'pointer';
  // Allow callers to override individual style properties
  if (fill !== undefined) {
    svg.innerHTML = dieMarkupCustom(value, fill, stroke, sw, '', pipOp);
  } else {
    svg.innerHTML = dieMarkup(value, state);
  }
  return svg;
}

/**
 * Return an SVG HTML string for a die face (for innerHTML patterns).
 * @param {number} value - Die value 1-6
 * @param {object} options
 * @param {number} options.size - Width/height in px (default 48)
 * @param {string} options.state - Visual state (default 'normal')
 * @returns {string}
 */
export function dieSVGString(value, options = {}) {
  const { size = 48, state = 'normal' } = options;
  return `<svg viewBox="0 0 48 48" width="${size}" height="${size}" class="die-svg">${dieMarkup(value, state)}</svg>`;
}

/**
 * Return a small inline SVG string for embedding in prose text.
 * Sized relative to surrounding text (~1.2em).
 * @param {number} value - Die value 1-6
 * @returns {string}
 */
export function inlineDieSVG(value) {
  const s = STATES.normal;
  let inner = `<rect x="1" y="1" width="46" height="46" rx="8" fill="${s.fill}" stroke="${s.stroke}" stroke-width="${s.sw}"/>`;
  (PIPS[value] || []).forEach(p => {
    inner += `<circle cx="${p.cx}" cy="${p.cy}" r="4.5" fill="var(--text)"/>`;
  });
  return `<svg viewBox="0 0 48 48" class="die-inline" style="width:1.2em;height:1.2em;vertical-align:-0.15em">${inner}</svg>`;
}
