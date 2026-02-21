import type { GameState, GameAction } from './types.ts';
import { gameReducer, initialState, saveState } from './reducer.ts';

type Listener = (state: GameState, prev: GameState, action: GameAction) => void;

let state: GameState = initialState();
const listeners = new Set<Listener>();

export function getState(): GameState {
  return state;
}

export function dispatch(action: GameAction): void {
  const prev = state;
  state = gameReducer(state, action);
  saveState(state);
  for (const fn of listeners) {
    fn(state, prev, action);
  }
}

export function subscribe(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function initStore(): void {
  state = initialState();
}
