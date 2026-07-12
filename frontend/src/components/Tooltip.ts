/**
 * Accessible guidance tooltips for the Play UI.
 *
 * One shared floating tooltip node (appended to <body>, position: fixed) is
 * reused for every term, so there is no per-term DOM clutter and no reflow of
 * the pixel-stable layout. Terms show on hover AND keyboard focus, hide on
 * leave / blur / Esc, and toggle on touch.
 */

let tipEl: HTMLDivElement | null = null;
let activeTerm: HTMLElement | null = null;

const SHARED_ID = 'ui-tooltip-shared';

function ensureTip(): HTMLDivElement {
  if (tipEl) return tipEl;
  const el = document.createElement('div');
  el.className = 'ui-tooltip';
  el.id = SHARED_ID;
  el.setAttribute('role', 'tooltip');
  el.style.display = 'none';
  document.body.appendChild(el);
  tipEl = el;
  return el;
}

/** Position the shared tooltip above `term`, flipping below and clamping to the viewport. */
function place(term: HTMLElement): void {
  const tip = ensureTip();
  const r = term.getBoundingClientRect();
  const margin = 8;
  const tw = tip.offsetWidth;
  const th = tip.offsetHeight;

  let left = r.left + r.width / 2 - tw / 2;
  left = Math.max(margin, Math.min(left, window.innerWidth - tw - margin));

  let top = r.top - th - margin;
  let below = false;
  if (top < margin) {
    top = r.bottom + margin;
    below = true;
  }
  tip.style.left = `${Math.round(left)}px`;
  tip.style.top = `${Math.round(top)}px`;
  tip.classList.toggle('below', below);
}

function show(term: HTMLElement, text: string): void {
  const tip = ensureTip();
  tip.textContent = text;
  // Reveal invisibly to measure, then position, then show.
  tip.style.visibility = 'hidden';
  tip.style.display = 'block';
  place(term);
  tip.style.visibility = 'visible';
  activeTerm = term;
}

function hide(): void {
  if (tipEl) tipEl.style.display = 'none';
  activeTerm = null;
}

/** Resolve a text argument that may be a literal string or a getter. */
function resolve(text: string | (() => string)): string {
  return typeof text === 'function' ? text() : text;
}

interface Options {
  /** Add the dotted-underline "defined term" affordance + help cursor. Default true. */
  underline?: boolean;
  /** Make the term keyboard-focusable and toggle on touch. Default true. */
  interactive?: boolean;
}

/**
 * Mark `el` as a term with a guidance tooltip.
 * @param el the element the user hovers/focuses
 * @param text tooltip copy, or a getter for dynamic content (e.g. a die's state)
 */
export function attachTooltip(
  el: HTMLElement,
  text: string | (() => string),
  opts: Options = {},
): void {
  const { underline = true, interactive = true } = opts;

  if (underline) el.classList.add('has-tooltip');
  ensureTip();
  if (interactive) {
    if (!el.hasAttribute('tabindex')) el.setAttribute('tabindex', '0');
    el.setAttribute('aria-describedby', SHARED_ID);
  }

  el.addEventListener('mouseenter', () => show(el, resolve(text)));
  el.addEventListener('mouseleave', hide);
  el.addEventListener('focus', () => show(el, resolve(text)));
  el.addEventListener('blur', hide);
  el.addEventListener('keydown', (e) => {
    if ((e as KeyboardEvent).key === 'Escape') hide();
  });

  if (interactive) {
    // Touch: toggle without hijacking a non-interactive element's default.
    el.addEventListener(
      'touchstart',
      (e) => {
        e.preventDefault();
        if (activeTerm === el) hide();
        else show(el, resolve(text));
      },
      { passive: false },
    );
  }
}

/**
 * Hover-only tooltip for an already-interactive element (e.g. a clickable die).
 * No underline, no cursor change, no touch hijack, no focus stop, so it never
 * fights the element's own tap/click behavior. Text is dynamic.
 */
export function attachHoverTooltip(el: HTMLElement, getText: () => string): void {
  ensureTip();
  el.addEventListener('mouseenter', () => {
    const t = getText();
    if (t) show(el, t);
  });
  el.addEventListener('mouseleave', hide);
}
