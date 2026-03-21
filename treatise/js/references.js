// Reference marker tooltips — hover over [N] superscripts to see the citation

let hideTimeout = null;

function createTooltip(target, html) {
  const rect = target.getBoundingClientRect();
  removeTooltips();
  if (hideTimeout) clearTimeout(hideTimeout);

  const tip = document.createElement('div');
  tip.className = 'annotation-tooltip';
  tip.innerHTML = html;
  document.body.appendChild(tip);

  // Open all links in tooltip in new tab
  tip.querySelectorAll('a[href^="http"]').forEach(a => { a.target = '_blank'; a.rel = 'noopener'; });

  tip.addEventListener('mouseenter', () => {
    if (hideTimeout) clearTimeout(hideTimeout);
  });
  tip.addEventListener('mouseleave', scheduleHide);

  const tipRect = tip.getBoundingClientRect();
  let left = rect.left + rect.width / 2 - tipRect.width / 2;
  let top = rect.top - tipRect.height - 6;

  if (left < 8) left = 8;
  if (left + tipRect.width > window.innerWidth - 8) left = window.innerWidth - tipRect.width - 8;
  if (top < 8) top = rect.bottom + 6;

  tip.style.left = `${left + window.scrollX}px`;
  tip.style.top = `${top + window.scrollY}px`;
}

function removeTooltips() {
  if (hideTimeout) clearTimeout(hideTimeout);
  document.querySelectorAll('.annotation-tooltip').forEach(el => el.remove());
}

function scheduleHide() {
  if (hideTimeout) clearTimeout(hideTimeout);
  hideTimeout = setTimeout(removeTooltips, 150);
}

export function initReferences() {
  document.querySelectorAll('a.ref-marker').forEach(el => {
    const refNum = el.dataset.ref;
    if (!refNum) return;

    const refAnchor = document.getElementById(`ref-${refNum}`);
    if (!refAnchor) return;

    const li = refAnchor.closest('li');
    if (!li) return;

    const clone = li.cloneNode(true);
    // Remove the anchor element itself from the clone
    const anchorEl = clone.querySelector(`a[id="ref-${refNum}"]`);
    if (anchorEl) {
      const parent = anchorEl.parentElement;
      if (parent && parent.tagName === 'P' && parent.children.length === 1 && parent.textContent.trim() === '') {
        parent.remove();
      } else {
        anchorEl.remove();
      }
    }

    const html = clone.innerHTML.trim();
    if (!html) return;

    el.addEventListener('mouseenter', () => createTooltip(el, html));
    el.addEventListener('mouseleave', scheduleHide);

    el.addEventListener('click', (e) => {
      e.preventDefault();
      if (document.querySelector('.annotation-tooltip')) {
        removeTooltips();
      } else {
        createTooltip(el, html);
      }
    });
  });

  // Dismiss on click elsewhere
  document.addEventListener('click', (e) => {
    if (!e.target.closest('a.ref-marker') && !e.target.closest('.annotation-tooltip')) {
      removeTooltips();
    }
  });
}
