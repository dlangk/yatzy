export function initTheme() {
  const toggle = document.getElementById('theme-toggle');
  const icon = document.getElementById('theme-icon');
  if (!toggle || !icon) return;

  function setTheme(dark) {
    document.documentElement.classList.toggle('dark', dark);
    icon.textContent = dark ? '\u2600' : '\u263E';
    localStorage.setItem('theme', dark ? 'dark' : 'light');
  }

  const saved = localStorage.getItem('theme');
  if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    setTheme(true);
  }

  toggle.addEventListener('click', () => {
    setTheme(!document.documentElement.classList.contains('dark'));
  });

  return { setTheme };
}
