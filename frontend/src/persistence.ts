/**
 * Single source of truth for what is saved and how.
 *
 * Two independent, versioned localStorage keys:
 *   - `yatzy-game-state`: the in-progress game (durable game data only). Cleared
 *     by New Game. Excludes derived/heavy fields (lastEvalResponse is a server
 *     response; undo/redo history is kept in memory only).
 *   - `yatzy-prefs`: user preferences (hints, debug, guide, autoplay speed).
 *     Independent of the game, so it survives New Game.
 *
 * Payloads are wrapped as { v, data }; a version bump drops incompatible old
 * data rather than trying to read a stale shape.
 */
import type { GameState, GameStateSnapshot, Prefs } from './types.ts';

const GAME_KEY = 'yatzy-game-state';
const PREFS_KEY = 'yatzy-prefs';
const VERSION = 2;

export const DEFAULT_PREFS: Prefs = {
  showHints: true,
  showDebug: false,
  guideOpen: true,
  autoplayDelay: 500,
};

/** The persisted game shape: the game snapshot minus the re-derivable eval response. */
export type PersistedGame = Omit<GameStateSnapshot, 'lastEvalResponse'>;

function read(key: string): unknown {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed?.v !== VERSION) return null; // incompatible / pre-versioning data
    return parsed.data;
  } catch {
    return null;
  }
}

function write(key: string, data: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify({ v: VERSION, data }));
  } catch {
    /* quota exceeded or private browsing — ignore */
  }
}

/** Persist the durable game data (excludes prefs, history, and lastEvalResponse). */
export function saveGame(state: GameState): void {
  const {
    lastEvalResponse: _l,
    undoStack: _u,
    redoStack: _r,
    prefs: _p,
    ...game
  } = state;
  void _l; void _u; void _r; void _p;
  write(GAME_KEY, game);
}

export function loadGame(): PersistedGame | null {
  return read(GAME_KEY) as PersistedGame | null;
}

export function clearGame(): void {
  try {
    localStorage.removeItem(GAME_KEY);
  } catch {
    /* ignore */
  }
}

export function savePrefs(prefs: Prefs): void {
  write(PREFS_KEY, prefs);
}

export function loadPrefs(): Prefs {
  const p = read(PREFS_KEY) as Partial<Prefs> | null;
  return { ...DEFAULT_PREFS, ...(p ?? {}) };
}
