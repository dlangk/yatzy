const cache = new Map();
let currentSlug = null;
let drawer, drawerTitle, drawerBody, backdrop;

function createDrawerDOM() {
  backdrop = document.createElement('div');
  backdrop.className = 'concept-drawer-backdrop';
  document.body.appendChild(backdrop);

  drawer = document.createElement('aside');
  drawer.className = 'concept-drawer';
  drawer.innerHTML = `
    <div class="concept-drawer-header">
      <span class="concept-drawer-title"></span>
      <button class="concept-drawer-close" aria-label="Close">&times;</button>
    </div>
    <div class="concept-drawer-body"></div>
  `;
  document.body.appendChild(drawer);

  drawerTitle = drawer.querySelector('.concept-drawer-title');
  drawerBody = drawer.querySelector('.concept-drawer-body');

  drawer.querySelector('.concept-drawer-close').addEventListener('click', closeDrawer);
  backdrop.addEventListener('click', closeDrawer);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeDrawer();
  });
}

function closeDrawer() {
  drawer.classList.remove('open');
  backdrop.classList.remove('open');
  document.querySelectorAll('.concept.active').forEach(el => el.classList.remove('active'));
  currentSlug = null;
}

async function openDrawer(slug, triggerEl) {
  if (currentSlug === slug) {
    closeDrawer();
    return;
  }

  // Clear previous highlight
  document.querySelectorAll('.concept.active').forEach(el => el.classList.remove('active'));
  triggerEl.classList.add('active');
  currentSlug = slug;

  // Fetch and cache
  if (!cache.has(slug)) {
    try {
      const resp = await fetch(`concepts/${slug}.md`);
      if (!resp.ok) throw new Error(`${resp.status}`);
      cache.set(slug, await resp.text());
    } catch (err) {
      cache.set(slug, `# Not Found\n\nConcept "${slug}" is not available yet.`);
    }
  }

  const md = cache.get(slug);
  const html = marked.parse(md);

  // Extract first h1 as title, remove from body
  const tmp = document.createElement('div');
  tmp.innerHTML = html;
  const h1 = tmp.querySelector('h1');
  if (h1) {
    drawerTitle.textContent = h1.textContent;
    h1.remove();
  } else {
    drawerTitle.textContent = slug.replace(/-/g, ' ');
  }
  drawerBody.innerHTML = tmp.innerHTML;

  drawer.classList.add('open');
  backdrop.classList.add('open');
  drawerBody.scrollTop = 0;
}

export function initConceptDrawer() {
  createDrawerDOM();

  document.getElementById('treatise-root').addEventListener('click', (e) => {
    const span = e.target.closest('.concept');
    if (!span) return;
    e.preventDefault();
    openDrawer(span.dataset.concept, span);
  });
}
