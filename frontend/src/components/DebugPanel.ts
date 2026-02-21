import { getState, dispatch, subscribe } from '../store.ts';

export function initDebugPanel(toggleContainer: HTMLElement, panelContainer: HTMLElement): void {
  const toggleBtn = document.createElement('button');
  toggleBtn.textContent = 'Show Debug';
  toggleBtn.addEventListener('click', () => dispatch({ type: 'TOGGLE_DEBUG' }));
  toggleContainer.appendChild(toggleBtn);

  panelContainer.className = 'debug-panel';
  panelContainer.style.display = 'none';

  const stateLabel = document.createElement('div');
  stateLabel.className = 'label';
  stateLabel.textContent = 'Game State';
  panelContainer.appendChild(stateLabel);

  const stateContent = document.createElement('div');
  panelContainer.appendChild(stateContent);

  const evalLabel = document.createElement('div');
  evalLabel.className = 'label';
  evalLabel.style.marginTop = '12px';
  evalLabel.textContent = 'Last Eval Response';
  panelContainer.appendChild(evalLabel);

  const evalContent = document.createElement('div');
  panelContainer.appendChild(evalContent);

  function render() {
    const s = getState();
    toggleBtn.textContent = s.showDebug ? 'Hide Debug' : 'Show Debug';
    panelContainer.style.display = s.showDebug ? 'block' : 'none';

    if (s.showDebug) {
      stateContent.textContent = JSON.stringify({
        dice: s.dice,
        upperScore: s.upperScore,
        scoredCategories: s.scoredCategories,
        rerollsRemaining: s.rerollsRemaining,
        totalScore: s.totalScore,
        bonus: s.bonus,
        turnPhase: s.turnPhase,
        sortMap: s.sortMap,
      }, null, 2);

      if (s.lastEvalResponse) {
        evalLabel.style.display = 'block';
        evalContent.style.display = 'block';
        evalContent.textContent = JSON.stringify(s.lastEvalResponse, null, 2);
      } else {
        evalLabel.style.display = 'none';
        evalContent.style.display = 'none';
      }
    }
  }

  render();
  subscribe(render);
}
