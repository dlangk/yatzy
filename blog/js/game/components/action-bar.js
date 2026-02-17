import { dispatch, subscribe, getState } from '../store.js';

export function initActionBar(container) {
  const bar = document.createElement('div');
  bar.className = 'game-action-bar';
  container.appendChild(bar);

  const mainBtn = document.createElement('button');
  mainBtn.className = 'game-btn-primary';

  const rerollControls = document.createElement('span');
  rerollControls.style.cssText = 'display:inline-flex;gap:4px;align-items:center';

  const minusBtn = document.createElement('button');
  minusBtn.className = 'game-btn-small';
  minusBtn.innerHTML = '&minus;';

  const plusBtn = document.createElement('button');
  plusBtn.className = 'game-btn-small';
  plusBtn.textContent = '+';

  rerollControls.append(minusBtn, plusBtn);

  const resetBtn = document.createElement('button');
  resetBtn.className = 'game-btn-secondary';
  resetBtn.textContent = 'Reset';

  const gameOverLabel = document.createElement('strong');
  gameOverLabel.textContent = 'Game Over!';
  gameOverLabel.style.display = 'none';

  bar.append(gameOverLabel, mainBtn, rerollControls, resetBtn);

  mainBtn.addEventListener('click', () => {
    const s = getState();
    if (s.turnPhase === 'idle') {
      dispatch({ type: 'ROLL' });
    } else if (s.turnPhase === 'rolled') {
      dispatch({ type: 'REROLL' });
    } else if (s.turnPhase === 'game_over') {
      dispatch({ type: 'RESET_GAME' });
    }
  });

  minusBtn.addEventListener('click', () => {
    const s = getState();
    dispatch({ type: 'SET_REROLLS', rerollsRemaining: s.rerollsRemaining - 1 });
  });

  plusBtn.addEventListener('click', () => {
    const s = getState();
    dispatch({ type: 'SET_REROLLS', rerollsRemaining: s.rerollsRemaining + 1 });
  });

  resetBtn.addEventListener('click', () => {
    if (window.confirm('Reset game? All progress will be lost.')) {
      dispatch({ type: 'RESET_GAME' });
    }
  });

  function render(state) {
    if (state.turnPhase === 'game_over') {
      gameOverLabel.style.display = '';
      mainBtn.textContent = 'New Game';
      mainBtn.disabled = false;
      rerollControls.style.visibility = 'hidden';
      resetBtn.style.visibility = 'hidden';
    } else if (state.turnPhase === 'idle') {
      gameOverLabel.style.display = 'none';
      mainBtn.textContent = 'Roll';
      mainBtn.disabled = false;
      rerollControls.style.visibility = 'hidden';
      resetBtn.style.visibility = 'visible';
    } else {
      gameOverLabel.style.display = 'none';
      mainBtn.textContent = `Reroll (${state.rerollsRemaining})`;
      mainBtn.disabled = state.rerollsRemaining <= 0;
      rerollControls.style.visibility = 'visible';
      minusBtn.disabled = state.rerollsRemaining <= 0;
      plusBtn.disabled = state.rerollsRemaining >= 2;
      resetBtn.style.visibility = 'visible';
    }
  }

  render(getState());
  subscribe((state) => render(state));
}
