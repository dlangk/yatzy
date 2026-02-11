export interface DieState {
  value: number;
  held: boolean;
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

export interface GameState {
  dice: DieState[];
  upperScore: number;
  scoredCategories: number;
  rerollsRemaining: number;
  categories: CategoryState[];
  totalScore: number;
  bonus: number;
  lastEvalResponse: EvaluateResponse | null;
  sortMap: number[] | null;
  showDebug: boolean;
  turnPhase: TurnPhase;
}

export type GameAction =
  | { type: 'ROLL' }
  | { type: 'TOGGLE_DIE'; index: number }
  | { type: 'REROLL' }
  | { type: 'SCORE_CATEGORY'; categoryId: number }
  | { type: 'SET_EVAL_RESPONSE'; response: EvaluateResponse; sortMap: number[] }
  | { type: 'TOGGLE_DEBUG' }
  | { type: 'RESET_GAME' }
  | { type: 'SET_DIE_VALUE'; index: number; value: number }
  | { type: 'SET_REROLLS'; rerollsRemaining: number }
  | { type: 'SET_CATEGORY_SCORE'; categoryId: number; score: number }
  | { type: 'UNSET_CATEGORY'; categoryId: number };
