import { getState, dispatch, subscribe } from '../store.ts';

export function initActionBar(container: HTMLElement): void {
  container.className = 'action-bar';

  const gameOverLabel = document.createElement('strong');
  gameOverLabel.textContent = 'Game Over!';

  const mainBtn = document.createElement('button');
  mainBtn.style.fontSize = '16px';
  mainBtn.style.padding = '6px 20px';

  const newGameBtn = document.createElement('button');
  newGameBtn.textContent = 'New Game';
  newGameBtn.style.fontSize = '16px';
  newGameBtn.style.padding = '6px 16px';
  newGameBtn.addEventListener('click', () => dispatch({ type: 'RESET_GAME' }));

  const rerollControls = document.createElement('span');
  rerollControls.className = 'reroll-controls';

  const minusBtn = document.createElement('button');
  minusBtn.className = 'small-btn';
  minusBtn.innerHTML = '&minus;';
  minusBtn.addEventListener('click', () => {
    const s = getState();
    dispatch({ type: 'SET_REROLLS', rerollsRemaining: s.rerollsRemaining - 1 });
  });

  const plusBtn = document.createElement('button');
  plusBtn.className = 'small-btn';
  plusBtn.textContent = '+';
  plusBtn.addEventListener('click', () => {
    const s = getState();
    dispatch({ type: 'SET_REROLLS', rerollsRemaining: s.rerollsRemaining + 1 });
  });

  rerollControls.appendChild(minusBtn);
  rerollControls.appendChild(plusBtn);

  const resetBtn = document.createElement('button');
  resetBtn.className = 'reset-btn';
  resetBtn.textContent = 'Reset';
  resetBtn.addEventListener('click', () => {
    if (window.confirm('Reset game? All progress will be lost.')) {
      dispatch({ type: 'RESET_GAME' });
    }
  });

  // Placeholder for game-over hidden reset (keeps layout stable)
  const hiddenPlaceholder = document.createElement('button');
  hiddenPlaceholder.style.fontSize = '12px';
  hiddenPlaceholder.style.padding = '4px 10px';
  hiddenPlaceholder.style.visibility = 'hidden';
  hiddenPlaceholder.textContent = 'Reset';

  mainBtn.addEventListener('click', () => {
    const s = getState();
    if (s.turnPhase === 'idle') {
      dispatch({ type: 'ROLL' });
    } else {
      dispatch({ type: 'REROLL' });
    }
  });

  function render() {
    const s = getState();
    container.innerHTML = '';

    if (s.turnPhase === 'game_over') {
      container.appendChild(gameOverLabel);
      container.appendChild(newGameBtn);
      container.appendChild(hiddenPlaceholder);
      return;
    }

    if (s.turnPhase === 'idle') {
      mainBtn.textContent = 'Roll';
      mainBtn.disabled = false;
    } else {
      mainBtn.textContent = `Reroll (${s.rerollsRemaining})`;
      mainBtn.disabled = s.rerollsRemaining <= 0;
    }
    container.appendChild(mainBtn);

    minusBtn.disabled = s.rerollsRemaining <= 0;
    plusBtn.disabled = s.rerollsRemaining >= 2;
    rerollControls.style.visibility = s.turnPhase === 'rolled' ? 'visible' : 'hidden';
    container.appendChild(rerollControls);

    container.appendChild(resetBtn);
  }

  render();
  subscribe(render);
}
