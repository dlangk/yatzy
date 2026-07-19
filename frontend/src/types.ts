export interface DieState {
  value: number;
  kept: boolean;
}

export interface TrajectoryPoint {
  index: number;
  turn: number;
  event: 'start' | 'roll' | 'reroll' | 'score';
  expectedFinal: number;
  accumulatedScore: number;
  stateEv: number;
  upperScore?: number;
  scoredCategories?: number;
  percentiles?: Record<string, number>;
  label?: string;
  delta?: number;
  /** Per-detent cost of this decision (index 0-4). deltas[riskLevel] is shown;
   *  delta above stays as the neutral fallback for points saved before the slider. */
  deltas?: number[];
}

export interface CategoryState {
  id: number;
  name: string;
  score: number;
  isScored: boolean;
  suggestedScore: number;
  evIfScored: number;
  available: boolean;
}

export interface EvaluateRequest {
  dice: number[];
  upper_score: number;
  scored_categories: number;
  rerolls_remaining: number;
}

export interface EvaluateResponse {
  mask_evs?: number[];
  optimal_mask?: number;
  optimal_mask_ev?: number;
  categories: {
    id: number;
    name: string;
    score: number;
    available: boolean;
    ev_if_scored: number;
  }[];
  optimal_category: number;
  optimal_category_ev: number;
  state_ev: number;
  /** Risk parameter for this response (0 = neutral). Present per detent. */
  theta?: number;
  /** All five risk detents (index order theta -0.08..+0.08). Flat fields above
   *  equal detents[2] (Neutral). Absent if the backend predates the slider. */
  detents?: EvaluateResponse[];
}

export type TurnPhase = 'idle' | 'rolled' | 'game_over';

/** Snapshot of game state at a turn boundary (excludes undo/redo stacks to avoid nesting). */
export interface GameStateSnapshot {
  dice: DieState[];
  upperScore: number;
  scoredCategories: number;
  rerollsRemaining: number;
  categories: CategoryState[];
  totalScore: number;
  bonus: number;
  /** Sum of all scored category values without bonus — used to avoid double-counting
   *  the terminal bonus that the solver's state_ev already includes. */
  rawScoredSum: number;
  lastEvalResponse: EvaluateResponse | null;
  /** Cached five-detent eval for the current roll (null between rolls). Derived,
   *  excluded from persistence like lastEvalResponse. */
  detents: EvaluateResponse[] | null;
  turnPhase: TurnPhase;
  trajectory: TrajectoryPoint[];
}

/** Durable user preferences. Kept separate from game data: persisted under its
 *  own key and preserved across New Game. Not part of undo/redo snapshots. */
export interface Prefs {
  showHints: boolean;
  showDebug: boolean;
  guideOpen: boolean;
  autoplayDelay: number;
  /** Override mode: reveals the manual game-state controls (die value arrows,
   *  reroll +/-, undo/redo). Off by default; these are "cheat" tools. */
  override: boolean;
  /** Selected risk detent, index 0-4 (2 = Neutral). Drives which detent slice
   *  the whole turn-by-turn feedback re-centers on. */
  riskLevel: number;
}

export interface GameState extends GameStateSnapshot {
  undoStack: GameStateSnapshot[];
  redoStack: GameStateSnapshot[];
  prefs: Prefs;
}

export type GameAction =
  | { type: 'ROLL' }
  | { type: 'TOGGLE_DIE'; index: number }
  | { type: 'REROLL' }
  | { type: 'SCORE_CATEGORY'; categoryId: number }
  | { type: 'SET_EVAL_RESPONSE'; response: EvaluateResponse; trajectoryEvent?: 'roll' | 'reroll'; rerollLabel?: string; rerollDeltas?: number[] }
  | { type: 'SET_RISK_LEVEL'; level: number }
  | { type: 'TOGGLE_DEBUG' }
  | { type: 'RESET_GAME' }
  | { type: 'SET_DIE_VALUE'; index: number; value: number }
  | { type: 'SET_REROLLS'; rerollsRemaining: number }
  | { type: 'SET_CATEGORY_SCORE'; categoryId: number; score: number }
  | { type: 'UNSET_CATEGORY'; categoryId: number }
  | { type: 'SET_INITIAL_EV'; ev: number }
  | { type: 'SET_DENSITY_RESULT'; index: number; percentiles: Record<string, number> }
  | { type: 'TOGGLE_HINTS' }
  | { type: 'TOGGLE_GUIDE' }
  | { type: 'TOGGLE_OVERRIDE' }
  | { type: 'SET_AUTOPLAY_DELAY'; delay: number }
  | { type: 'UNDO' }
  | { type: 'REDO' };

/** Read a trajectory point's cost at a given detent, falling back to the single
 *  neutral delta for points saved before per-detent deltas existed. */
export function pointDelta(p: TrajectoryPoint, level: number): number | null {
  return p.deltas?.[level] ?? p.delta ?? null;
}
