import { getState, dispatch, subscribe } from '../store.ts';

/** Render the action bar: Roll/Reroll button, reroll counter, hints toggle, and reset. */
export function initActionBar(container: HTMLElement): void {
  container.className = 'action-bar';

  const gameOverLabel = document.createElement('strong');
  gameOverLabel.textContent = 'Game Over!';

  const mainBtn = document.createElement('button');

  const newGameBtn = document.createElement('button');
  newGameBtn.className = 'game-btn-primary';
  newGameBtn.textContent = 'New Game';
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

  const hintsBtn = document.createElement('button');
  hintsBtn.className = 'game-btn-secondary';
  hintsBtn.addEventListener('click', () => dispatch({ type: 'TOGGLE_HINTS' }));

  const undoBtn = document.createElement('button');
  undoBtn.className = 'game-btn-secondary';
  undoBtn.textContent = 'Undo';
  undoBtn.addEventListener('click', () => dispatch({ type: 'UNDO' }));

  const redoBtn = document.createElement('button');
  redoBtn.className = 'game-btn-secondary';
  redoBtn.textContent = 'Redo';
  redoBtn.addEventListener('click', () => dispatch({ type: 'REDO' }));

  const resetBtn = document.createElement('button');
  resetBtn.className = 'game-btn-secondary';
  resetBtn.textContent = 'Reset';
  resetBtn.addEventListener('click', () => {
    if (window.confirm('Reset game? All progress will be lost.')) {
      dispatch({ type: 'RESET_GAME' });
    }
  });

  // Placeholder for game-over hidden reset (keeps layout stable)
  const hiddenPlaceholder = document.createElement('button');
  hiddenPlaceholder.className = 'game-btn-secondary';
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
      mainBtn.className = 'game-btn-primary';
    } else if (s.rerollsRemaining > 0) {
      mainBtn.textContent = `Reroll (${s.rerollsRemaining})`;
      mainBtn.disabled = false;
      mainBtn.className = 'game-btn-primary';
    } else {
      mainBtn.textContent = `Reroll (0)`;
      mainBtn.disabled = true;
      mainBtn.className = 'game-btn-primary';
    }
    container.appendChild(mainBtn);

    const rerollActive = s.turnPhase === 'rolled';
    minusBtn.disabled = !rerollActive || s.rerollsRemaining <= 0;
    plusBtn.disabled = !rerollActive || s.rerollsRemaining >= 2;
    rerollControls.style.opacity = rerollActive ? '1' : '0.3';
    container.appendChild(rerollControls);

    hintsBtn.textContent = s.showHints ? 'Hide Hints' : 'Show Hints';
    container.appendChild(hintsBtn);

    undoBtn.disabled = s.undoStack.length === 0;
    container.appendChild(undoBtn);

    redoBtn.disabled = s.redoStack.length === 0;
    container.appendChild(redoBtn);

    container.appendChild(resetBtn);
  }

  render();
  subscribe(render);
}
