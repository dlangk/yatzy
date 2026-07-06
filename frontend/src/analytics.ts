// Play-app analytics. Fires GA4 events via window.yatzyTrack (set up by
// shared/nav.js, which no-ops off production). Subscribed to the store so the
// reducer stays pure; every meaningful transition maps to one event.
import { subscribe } from './store.ts';
import type { GameState, GameAction } from './types.ts';

declare global {
  interface Window {
    yatzyTrack?: (name: string, params?: Record<string, unknown>) => void;
  }
}

export function initPlayAnalytics(): void {
  subscribe((state: GameState, prev: GameState, action: GameAction) => {
    const track = window.yatzyTrack;
    if (!track) return;

    switch (action.type) {
      case 'ROLL':
        // Turn-start roll of a fresh scorecard = a new game.
        if (prev.scoredCategories === 0 && prev.turnPhase === 'idle') {
          track('game_start');
        }
        break;

      case 'REROLL':
        track('reroll', { rerolls_left: state.rerollsRemaining });
        break;

      case 'SCORE_CATEGORY': {
        const cat = state.categories[action.categoryId];
        if (cat) track('category_scored', { category: cat.name, points: cat.score });
        if (state.turnPhase === 'game_over' && prev.turnPhase !== 'game_over') {
          track('game_complete', {
            score: state.totalScore,
            upper: state.upperScore,
            bonus: state.bonus,
          });
        }
        break;
      }
    }
  });
}
