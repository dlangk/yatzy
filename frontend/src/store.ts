import type { GameState, GameAction } from './types.ts';
import { gameReducer, initialState, saveState } from './reducer.ts';

type Listener = (state: GameState, prev: GameState, action: GameAction) => void;

let state: GameState = initialState();
const listeners = new Set<Listener>();

/** Returns a readonly snapshot of the current game state. */
export function getState(): GameState {
  return state;
}

/** Dispatches an action to the reducer, updates state, persists to localStorage, and notifies all subscribers. */
export function dispatch(action: GameAction): void {
  const prev = state;
  state = gameReducer(state, action);
  saveState(state);
  for (const fn of listeners) {
    fn(state, prev, action);
  }
}

/** Registers a listener called on every state change. Returns an unsubscribe function. */
export function subscribe(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function initStore(): void {
  state = initialState();
}
