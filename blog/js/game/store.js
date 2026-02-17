import { gameReducer, initialState, saveState } from './reducer.js';

let state = initialState();
const listeners = new Set();

export function getState() { return state; }

export function dispatch(action) {
  const prev = state;
  state = gameReducer(state, action);
  saveState(state);
  if (state !== prev) listeners.forEach(fn => fn(state, prev, action));
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
