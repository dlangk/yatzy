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
  showDebug: boolean;
  turnPhase: TurnPhase;
  trajectory: TrajectoryPoint[];
  showHints: boolean;
}

export interface GameState extends GameStateSnapshot {
  undoStack: GameStateSnapshot[];
  redoStack: GameStateSnapshot[];
}

export type GameAction =
  | { type: 'ROLL' }
  | { type: 'TOGGLE_DIE'; index: number }
  | { type: 'REROLL' }
  | { type: 'SCORE_CATEGORY'; categoryId: number }
  | { type: 'SET_EVAL_RESPONSE'; response: EvaluateResponse; trajectoryEvent?: 'roll' | 'reroll'; rerollLabel?: string; rerollDelta?: number }
  | { type: 'TOGGLE_DEBUG' }
  | { type: 'RESET_GAME' }
  | { type: 'SET_DIE_VALUE'; index: number; value: number }
  | { type: 'SET_REROLLS'; rerollsRemaining: number }
  | { type: 'SET_CATEGORY_SCORE'; categoryId: number; score: number }
  | { type: 'UNSET_CATEGORY'; categoryId: number }
  | { type: 'SET_INITIAL_EV'; ev: number }
  | { type: 'SET_DENSITY_RESULT'; index: number; percentiles: Record<string, number> }
  | { type: 'TOGGLE_HINTS' }
  | { type: 'UNDO' }
  | { type: 'REDO' };
