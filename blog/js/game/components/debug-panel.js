import { subscribe, getState } from '../store.js';

export function initDebugPanel(container) {
  const wrapper = document.createElement('div');
  wrapper.style.textAlign = 'center';
  wrapper.style.marginTop = '12px';
  container.appendChild(wrapper);

  const toggleBtn = document.createElement('button');
  toggleBtn.className = 'game-btn-secondary';
  toggleBtn.textContent = 'Show Debug';
  wrapper.appendChild(toggleBtn);

  const panel = document.createElement('div');
  panel.className = 'game-debug-panel';
  panel.style.display = 'none';
  container.appendChild(panel);

  const stateLabel = document.createElement('div');
  stateLabel.className = 'debug-label';
  stateLabel.textContent = 'Game State';
  const stateContent = document.createElement('div');

  const evalLabel = document.createElement('div');
  evalLabel.className = 'debug-label';
  evalLabel.style.marginTop = '12px';
  evalLabel.textContent = 'Last Eval Response';
  const evalContent = document.createElement('div');

  panel.append(stateLabel, stateContent, evalLabel, evalContent);

  toggleBtn.addEventListener('click', () => {
    import('../store.js').then(({ dispatch }) => {
      dispatch({ type: 'TOGGLE_DEBUG' });
    });
  });

  function render(state) {
    toggleBtn.textContent = state.showDebug ? 'Hide Debug' : 'Show Debug';
    panel.style.display = state.showDebug ? '' : 'none';

    if (state.showDebug) {
      stateContent.textContent = JSON.stringify({
        dice: state.dice,
        upperScore: state.upperScore,
        scoredCategories: state.scoredCategories,
        rerollsRemaining: state.rerollsRemaining,
        totalScore: state.totalScore,
        bonus: state.bonus,
        turnPhase: state.turnPhase,
        sortMap: state.sortMap,
      }, null, 2);

      evalLabel.style.display = state.lastEvalResponse ? '' : 'none';
      evalContent.style.display = state.lastEvalResponse ? '' : 'none';
      if (state.lastEvalResponse) {
        evalContent.textContent = JSON.stringify(state.lastEvalResponse, null, 2);
      }
    }
  }

  render(getState());
  subscribe((state) => render(state));
}
