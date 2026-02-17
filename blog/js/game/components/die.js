import { dispatch } from '../store.js';

export function createDie(index) {
  const wrapper = document.createElement('div');
  wrapper.className = 'die-wrapper';

  const upBtn = document.createElement('button');
  upBtn.className = 'die-arrow';
  upBtn.innerHTML = '&#9650;';

  const btn = document.createElement('button');
  btn.className = 'die--interactive die--held die--faded';
  btn.textContent = '?';
  btn.title = 'Held (click to reroll)';

  const downBtn = document.createElement('button');
  downBtn.className = 'die-arrow';
  downBtn.innerHTML = '&#9660;';

  btn.addEventListener('click', () => dispatch({ type: 'TOGGLE_DIE', index }));

  upBtn.addEventListener('click', () => {
    const v = btn._value || 0;
    const active = btn._active;
    const upValue = (v >= 6 || v === 0) ? 1 : v + 1;
    dispatch({ type: 'SET_DIE_VALUE', index, value: upValue });
  });

  downBtn.addEventListener('click', () => {
    const v = btn._value || 0;
    const downValue = (v <= 1 || v === 0) ? 6 : v - 1;
    dispatch({ type: 'SET_DIE_VALUE', index, value: downValue });
  });

  wrapper.append(upBtn, btn, downBtn);

  return {
    el: wrapper,
    update(die, active, canToggle, showMaskHints, optimalMask) {
      const v = active ? die.value : 0;
      btn._value = v;
      btn._active = active;
      btn.textContent = v === 0 ? '?' : v;
      btn.disabled = !canToggle;

      // Reset classes
      btn.className = 'die--interactive';
      if (active) {
        btn.classList.add(die.held ? 'die--held' : 'die--will-reroll');
        btn.title = die.held ? 'Held (click to reroll)' : 'Will reroll (click to hold)';
      } else {
        btn.classList.add('die--held', 'die--faded');
        btn.title = '';
      }

      if (showMaskHints && optimalMask !== null) {
        if (optimalMask & (1 << index)) {
          btn.classList.add('die--optimal-reroll');
        } else {
          btn.classList.add('die--optimal-keep');
        }
      }
    },
  };
}
