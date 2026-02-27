/**
 * Progressive disclosure controller.
 * Injects inline toggle buttons before each .math / .code-detail block.
 * Blocks start collapsed via CSS; JS adds .layer-open to expand.
 */
export function initDepthToggle() {
  document.querySelectorAll('.math, .code-detail').forEach(block => {
    const isCode = block.classList.contains('code-detail');
    const label = isCode ? 'code' : 'math';

    const btn = document.createElement('button');
    btn.className = 'layer-toggle';
    btn.dataset.layer = isCode ? '3' : '2';
    btn.innerHTML = `<span class="toggle-arrow">&#9654;</span> ${label}`;

    btn.addEventListener('click', () => {
      if (block.classList.contains('layer-open')) {
        block.style.height = block.scrollHeight + 'px';
        block.offsetHeight;
        block.style.height = '0px';
        block.classList.remove('layer-open');
        btn.classList.remove('expanded');
      } else {
        block.style.height = 'auto';
        block.classList.add('layer-open');
        const h = block.scrollHeight;
        block.style.height = '0px';
        block.offsetHeight;
        block.style.height = h + 'px';
        btn.classList.add('expanded');
        const onEnd = () => {
          block.style.height = 'auto';
          block.removeEventListener('transitionend', onEnd);
        };
        block.addEventListener('transitionend', onEnd);
      }
    });

    block.parentNode.insertBefore(btn, block);
  });
}
