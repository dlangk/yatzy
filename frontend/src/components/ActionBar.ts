import { getState, dispatch, subscribe } from '../store.ts';

/** Render the action bar: Roll/Reroll button, reroll counter, hints toggle, and reset. */
export function initActionBar(container: HTMLElement): void {
  container.className = 'action-bar';

  const mainBtn = document.createElement('button');

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

  mainBtn.addEventListener('click', () => {
    const s = getState();
    if (s.turnPhase === 'game_over') {
      dispatch({ type: 'RESET_GAME' });
    } else if (s.turnPhase === 'idle') {
      dispatch({ type: 'ROLL' });
    } else {
      dispatch({ type: 'REROLL' });
    }
  });

  // Append children once — update in place on state changes
  container.appendChild(mainBtn);
  container.appendChild(rerollControls);
  container.appendChild(hintsBtn);
  container.appendChild(undoBtn);
  container.appendChild(redoBtn);
  container.appendChild(resetBtn);

  function render() {
    const s = getState();
    const isOver = s.turnPhase === 'game_over';

    if (isOver) {
      mainBtn.textContent = 'New Game';
      mainBtn.disabled = false;
      mainBtn.className = 'game-btn-primary';
    } else if (s.turnPhase === 'idle') {
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

    minusBtn.disabled = isOver || s.turnPhase !== 'rolled' || s.rerollsRemaining <= 0;
    plusBtn.disabled = isOver || s.turnPhase !== 'rolled' || s.rerollsRemaining >= 2;
    rerollControls.style.opacity = (!isOver && s.turnPhase === 'rolled') ? '1' : '0.3';

    hintsBtn.textContent = s.showHints ? 'Hide Hints' : 'Show Hints';
    hintsBtn.disabled = isOver;

    undoBtn.disabled = s.undoStack.length === 0;
    redoBtn.disabled = s.redoStack.length === 0;
    resetBtn.disabled = isOver;
  }

  render();
  subscribe((state, prev) => {
    if (state.turnPhase === prev.turnPhase &&
        state.rerollsRemaining === prev.rerollsRemaining &&
        state.showHints === prev.showHints &&
        state.undoStack === prev.undoStack &&
        state.redoStack === prev.redoStack) return;
    render();
  });
}
