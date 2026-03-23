/**
 * Shared navigation bar for all Yatzy UIs.
 * Self-contained: injects its own HTML + CSS, handles theme toggle.
 * Include via <script type="module" src="/yatzy/shared/nav.js"></script>
 */

const PAGES = [
  { id: 'treatise', label: 'Treatise', href: '/yatzy/', tooltip: 'The math behind optimal Yatzy play' },
  { id: 'play',     label: 'Play',     href: '/yatzy/play/', tooltip: 'Play Yatzy with real-time optimal hints' },
  { id: 'profile',  label: 'Profile',  href: '/yatzy/profile/', tooltip: 'Discover your strategic personality' },
];

function detectActive() {
  const path = location.pathname;
  if (path.includes('/play')) return 'play';
  if (path.includes('/profile')) return 'profile';
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
      padding: 2px 12px;
      border-radius: 5px;
      line-height: 20px;
      height: 24px;
      transition: background 0.15s, color 0.15s, box-shadow 0.15s;
      position: relative;
      width: 80px;
      text-align: center;
      font-weight: 600;
      border: 1px solid var(--border, #d4d3cd);
      background: var(--bg, #f6f5ef);
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

injectStyles();
injectNav();
