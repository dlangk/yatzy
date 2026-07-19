import { getState, dispatch, subscribe } from '../store.ts';

/**
 * Render the action bar as two rows:
 *   Row 1 (play): Roll/Reroll/New Game, and Reset. When Override mode is on it
 *     also shows the manual controls: reroll +/- and undo/redo.
 *   Row 2 (modes): Optimal hints, Intro guide, Override mode.
 *
 * Override mode is a "cheat" toggle: it reveals the controls that let you edit
 * the game state by hand (die values, reroll count, undo/redo). Those controls,
 * and the die value arrows in the dice bar, are hidden until it is on. Rather
 * than thread the flag through each component, we toggle a `override-mode` class
 * on <body> and let CSS reveal the `.override-only` controls.
 */
export function initActionBar(container: HTMLElement): void {
  container.className = 'action-bar';

  const playRow = document.createElement('div');
  playRow.className = 'action-row';
  const modeRow = document.createElement('div');
  modeRow.className = 'action-row';
  container.appendChild(playRow);
  container.appendChild(modeRow);

  const mainBtn = document.createElement('button');

  // Reroll counter +/- : manual override, hidden unless Override mode is on.
  const rerollControls = document.createElement('span');
  rerollControls.className = 'reroll-controls override-only';

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

  const undoBtn = document.createElement('button');
  undoBtn.className = 'game-btn-secondary override-only';
  undoBtn.textContent = 'Undo';
  undoBtn.addEventListener('click', () => dispatch({ type: 'UNDO' }));

  const redoBtn = document.createElement('button');
  redoBtn.className = 'game-btn-secondary override-only';
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

  // Optimal hints: compact toggle. Same label in both states; the accent colour
  // (via .active) signals on/off rather than a text change.
  const hintsBtn = document.createElement('button');
  hintsBtn.className = 'game-btn-secondary btn-toggle';
  hintsBtn.textContent = 'Optimal hints';
  hintsBtn.addEventListener('click', () => dispatch({ type: 'TOGGLE_HINTS' }));

  // Intro guide: toggles the "how to play" box (lives above the buttons).
  const guideBtn = document.createElement('button');
  guideBtn.className = 'game-btn-secondary btn-toggle';
  guideBtn.textContent = 'Intro guide';
  guideBtn.addEventListener('click', () => dispatch({ type: 'TOGGLE_GUIDE' }));

  // Override mode: reveals the manual game-state controls.
  const overrideBtn = document.createElement('button');
  overrideBtn.className = 'game-btn-secondary btn-toggle';
  overrideBtn.textContent = 'Override mode';
  overrideBtn.addEventListener('click', () => dispatch({ type: 'TOGGLE_OVERRIDE' }));

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

  // Row 1: play + (override) manual controls; Reset stays at the end.
  playRow.appendChild(mainBtn);
  playRow.appendChild(rerollControls);
  playRow.appendChild(undoBtn);
  playRow.appendChild(redoBtn);
  playRow.appendChild(resetBtn);

  // Row 2: mode toggles.
  modeRow.appendChild(hintsBtn);
  modeRow.appendChild(guideBtn);
  modeRow.appendChild(overrideBtn);

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

    hintsBtn.classList.toggle('active', s.prefs.showHints);
    // Hints stays toggleable even at game over, so you can review the final
    // board with or without the optimal highlights.
    hintsBtn.disabled = false;
    guideBtn.classList.toggle('active', s.prefs.guideOpen);
    overrideBtn.classList.toggle('active', s.prefs.override);

    // CSS reveals .override-only controls (here and the dice-bar arrows) off this.
    document.body.classList.toggle('override-mode', s.prefs.override);

    undoBtn.disabled = s.undoStack.length === 0;
    redoBtn.disabled = s.redoStack.length === 0;
    resetBtn.disabled = isOver;
  }

  render();
  subscribe((state, prev) => {
    if (state.turnPhase === prev.turnPhase &&
        state.rerollsRemaining === prev.rerollsRemaining &&
        state.prefs.showHints === prev.prefs.showHints &&
        state.prefs.guideOpen === prev.prefs.guideOpen &&
        state.prefs.override === prev.prefs.override &&
        state.undoStack === prev.undoStack &&
        state.redoStack === prev.redoStack) return;
    render();
  });
}
