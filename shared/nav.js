/**
 * Shared navigation bar for all Yatzy UIs.
 * Self-contained: injects its own HTML + CSS, handles theme toggle.
 * Include via <script type="module" src="/yatzy/shared/nav.js"></script>
 */

const PAGES = [
  { id: 'treatise', label: 'Treatise', href: '/yatzy/', tooltip: 'The math behind optimal Yatzy play' },
  { id: 'play',     label: 'Play',     href: '/yatzy/play/', tooltip: 'Play Yatzy with real-time optimal hints' },
  { id: 'prob',     label: 'Probabilities', href: '/yatzy/probabilities/', tooltip: 'Dice probabilities across two rerolls' },
  { id: 'profile',  label: 'Profile',  href: '/yatzy/profile/', tooltip: 'Discover your strategic personality' },
];

function detectActive() {
  const path = location.pathname;
  if (path.includes('/play')) return 'play';
  if (path.includes('/profile')) return 'profile';
  if (path.includes('/probabilities')) return 'prob';
  return 'treatise';
}

function injectStyles() {
  const style = document.createElement('style');
  style.textContent = `
    .site-nav {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      z-index: 9999;
      height: 36px;
      background: var(--bg-alt, #eae9e3);
      border-bottom: 1px solid var(--border, #d4d3cd);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      font-family: 'Newsreader', Georgia, 'Times New Roman', serif;
      font-size: 15px;
    }

    .site-nav-links {
      display: flex;
      gap: 4px;
      align-items: center;
    }

    .site-nav a {
      text-decoration: none;
      color: var(--text-muted, #555);
      /* top 4px: the serif labels have no descenders, so the font's empty
         descender reserve otherwise makes the text look high in the pill.
         With border-box + fixed height, flex re-centers, so the net downward
         shift is ~half the padding; 4px lands the ink on the pill's center. */
      padding: 4px 12px 0;
      border-radius: 5px;
      font-size: 15px;
      display: inline-flex;
      align-items: center;
      height: 24px;
      box-sizing: border-box;
      transition: background 0.15s, color 0.15s, box-shadow 0.15s, border-color 0.15s;
      position: relative;
      min-width: 80px;
      justify-content: center;
      font-weight: 600;
      border: 1px solid var(--border, #d4d3cd);
      background: var(--bg, #f6f5ef);
    }

    .site-nav a.site-nav-home {
      width: auto;
      font-size: 13px;
      font-weight: 400;
      padding: 4px 10px 0;
    }

    .site-nav a:hover {
      border-color: var(--accent, rgba(243, 112, 33, 0.8));
      color: var(--text, #050505);
    }

    .site-nav a.active {
      color: var(--accent, rgba(243, 112, 33, 0.8));
      background: var(--bg-alt, #eae9e3);
      border-color: var(--accent, rgba(243, 112, 33, 0.8));
      box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.12);
    }

    .site-nav-tooltip {
      position: absolute;
      top: 100%;
      left: 0;
      margin-top: 4px;
      padding: 5px 10px;
      background: var(--bg, #f6f5ef);
      border: 1px solid var(--border, #d4d3cd);
      border-radius: 5px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      font-size: 13px;
      color: var(--text, #050505);
      white-space: nowrap;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.15s;
      z-index: 10001;
    }

    .dark .site-nav-tooltip {
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
    }

    .site-nav a:hover .site-nav-tooltip {
      opacity: 1;
    }

    .site-nav-theme {
      background: none;
      border: 1px solid var(--border, #d4d3cd);
      border-radius: 50%;
      width: 28px;
      height: 28px;
      cursor: pointer;
      font-size: 14px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--text-muted, #555);
      transition: border-color 0.15s, color 0.15s;
    }

    .site-nav-theme:hover {
      border-color: var(--accent, rgba(243, 112, 33, 0.8));
      color: var(--accent, rgba(243, 112, 33, 0.8));
    }

    /* Push page content below the fixed nav */
    body { padding-top: 36px !important; }
  `;
  document.head.appendChild(style);
}

function injectNav() {
  const active = detectActive();
  const nav = document.createElement('nav');
  nav.className = 'site-nav';

  const links = document.createElement('div');
  links.className = 'site-nav-links';

  // Back to langkilde.se
  const home = document.createElement('a');
  home.href = 'https://langkilde.se';
  home.textContent = '← langkilde.se';
  home.className = 'site-nav-home';
  links.appendChild(home);

  for (const page of PAGES) {
    const a = document.createElement('a');
    a.href = page.href;
    a.textContent = page.label;
    if (page.id === active) a.classList.add('active');

    const tip = document.createElement('span');
    tip.className = 'site-nav-tooltip';
    tip.textContent = page.tooltip;
    a.appendChild(tip);

    links.appendChild(a);
  }

  const themeBtn = document.createElement('button');
  themeBtn.className = 'site-nav-theme';
  themeBtn.setAttribute('aria-label', 'Toggle dark mode');

  function updateIcon() {
    themeBtn.textContent = document.documentElement.classList.contains('dark') ? '\u2600' : '\u263E';
  }

  // Apply saved theme immediately
  const saved = localStorage.getItem('theme');
  if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.classList.add('dark');
  }
  updateIcon();

  themeBtn.addEventListener('click', () => {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    updateIcon();
    // Update any existing theme icon (for UIs that have their own toggle)
    const oldIcon = document.getElementById('theme-icon');
    if (oldIcon) oldIcon.textContent = isDark ? '\u2600' : '\u263E';
  });

  nav.appendChild(links);
  nav.appendChild(themeBtn);
  document.body.prepend(nav);

  // Hide the old standalone theme toggle if present
  const oldToggle = document.getElementById('theme-toggle');
  if (oldToggle) oldToggle.style.display = 'none';
}

// ── Analytics (GA4, dedicated "Yatzy" property) ────────────────────────────
// Mirrors langkilde.se's Analytics.astro: gtag.js (~150 KB) is deferred until
// the first real interaction (or a 10s fallback) so it stays out of the
// Lighthouse trace window and off bots, and never loads on localhost. All
// three UIs (treatise, play, profile) load this file, so this is the single
// place GA is wired up. `app` is set as a default param on every hit
// (page_view + custom events) so the property's reports can be sliced per UI.
const GA_ID = 'G-EKNV4GY2ZN';

function initAnalytics() {
  // Dev/preview guard: nav.js is plain JS with no build-time env, so unlike
  // the blog's `import.meta.env.PROD` gate we allowlist the production host.
  // This blocks the dev server on every form (localhost, 127.0.0.1, 0.0.0.0,
  // a LAN IP via `vite --host`, any preview host) — a denylist would leak
  // those. Off-production, keep a no-op yatzyTrack so callers need no env check.
  const host = location.hostname;
  const isProd = host === 'langkilde.se' || host.endsWith('.langkilde.se');
  if (!isProd) {
    window.yatzyTrack = window.yatzyTrack || function () {};
    return;
  }

  // Define dataLayer + gtag and queue js/config immediately (cheap, no
  // network) so custom events fired before gtag.js loads are correctly
  // ordered after config. Only the ~150 KB script is deferred.
  window.dataLayer = window.dataLayer || [];
  function gtag() { window.dataLayer.push(arguments); }
  window.gtag = window.gtag || gtag;

  const app = detectActive(); // 'treatise' | 'play' | 'profile'
  const path = location.pathname.replace(/\/+$/, '') || '/';

  window.gtag('js', new Date());
  window.gtag('set', { app: app }); // default param on all subsequent events
  window.gtag('config', GA_ID, {
    page_path: path,
    // Strip hash so TOC/footnote clicks don't spawn extra page_views.
    page_location: location.origin + path,
  });

  // Public helper for the three UIs: window.yatzyTrack('event_name', {params}).
  // Call it defensively, e.g. window.yatzyTrack?.('game_start', {...}).
  window.yatzyTrack = function (name, params) {
    window.gtag('event', name, Object.assign({ app: app }, params || {}));
  };

  let gaLoaded = false;
  function loadGA() {
    if (gaLoaded) return;
    gaLoaded = true;
    const script = document.createElement('script');
    script.async = true;
    script.src = 'https://www.googletagmanager.com/gtag/js?id=' + GA_ID;
    document.head.appendChild(script);
  }

  function onFirstInteraction() {
    if (gaLoaded) return;
    if ('requestIdleCallback' in window) requestIdleCallback(loadGA, { timeout: 2000 });
    else setTimeout(loadGA, 0);
  }
  ['pointerdown', 'scroll', 'keydown', 'touchstart'].forEach(function (type) {
    window.addEventListener(type, onFirstInteraction, { once: true, passive: true, capture: true });
  });
  // Count visitors who never interact, well after page quiescence.
  window.addEventListener('load', function () { setTimeout(loadGA, 10000); });
}

initAnalytics();
injectStyles();
injectNav();
