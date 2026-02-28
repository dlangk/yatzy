import type { GameState, GameAction, CategoryState, TrajectoryPoint } from './types.ts';
import { CATEGORY_NAMES, CATEGORY_COUNT, TOTAL_DICE, UPPER_SCORE_CAP, UPPER_BONUS, UPPER_CATEGORIES } from './constants.ts';

const STORAGE_KEY = 'yatzy-game-state';

/**
 * Persist game state to localStorage on every dispatch.
 *
 * Persisted fields: dice, scores, categories, trajectory, bonus,
 * turnPhase, rerollsRemaining, showDebug, showHints, lastEvalResponse, sortMap.
 *
 * Excluded (transient): pendingTrajectoryEvent, pendingRerollLabel,
 * pendingRerollDelta — these are mid-dispatch intermediates that are
 * consumed before the next render and must not survive a page reload.
 */
export function saveState(state: GameState): void {
  try {
    const { pendingTrajectoryEvent: _p, pendingRerollLabel: _l, pendingRerollDelta: _d, ...persistable } = state;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(persistable));
  } catch { /* quota exceeded or private browsing — ignore */ }
}

/**
 * Restore game state from localStorage on startup.
 *
 * Transient fields are nulled (pendingTrajectoryEvent, pendingRerollLabel,
 * pendingRerollDelta). Fields added after initial release are defaulted:
 * trajectory→[], showHints→true, lastEvalResponse→null, sortMap→null.
 */
function loadState(): GameState | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return { ...parsed, lastEvalResponse: parsed.lastEvalResponse ?? null, sortMap: parsed.sortMap ?? null, pendingTrajectoryEvent: null, pendingRerollLabel: null, pendingRerollDelta: null, trajectory: parsed.trajectory ?? [], showHints: parsed.showHints ?? true };
  } catch {
    return null;
  }
}

export function clearSavedState(): void {
  try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
}

function randomDie(): number {
  return Math.floor(Math.random() * 6) + 1;
}

function buildInitialCategories(): CategoryState[] {
  return CATEGORY_NAMES.map((name, id) => ({
    id,
    name,
    score: 0,
    isScored: false,
    suggestedScore: 0,
    evIfScored: 0,
    available: true,
  }));
}

function freshState(): GameState {
  return {
    dice: Array.from({ length: TOTAL_DICE }, () => ({ value: 0, held: false })),
    upperScore: 0,
    scoredCategories: 0,
    rerollsRemaining: 0,
    categories: buildInitialCategories(),
    totalScore: 0,
    bonus: 0,
    lastEvalResponse: null,
    sortMap: null,
    showDebug: false,
    turnPhase: 'idle',
    trajectory: [],
    pendingTrajectoryEvent: null,
    pendingRerollLabel: null,
    pendingRerollDelta: null,
    showHints: true,
  };
}

export function initialState(): GameState {
  return loadState() ?? freshState();
}

function computeUpperScore(categories: CategoryState[]): number {
  let sum = 0;
  for (let i = 0; i < UPPER_CATEGORIES; i++) {
    if (categories[i].isScored) {
      sum += categories[i].score;
    }
  }
  return Math.min(sum, UPPER_SCORE_CAP);
}

function computeTotalScore(categories: CategoryState[], bonus: number): number {
  let sum = 0;
  for (const cat of categories) {
    if (cat.isScored) {
      sum += cat.score;
    }
  }
  return sum + bonus;
}

function computeScoredMask(categories: CategoryState[]): number {
  return categories.reduce(
    (mask, c) => (c.isScored ? mask | (1 << c.id) : mask), 0
  );
}

function recomputeDerived(categories: CategoryState[]) {
  const upperScore = computeUpperScore(categories);
  const rawUpperSum = categories.slice(0, UPPER_CATEGORIES).reduce(
    (s, c) => s + (c.isScored ? c.score : 0), 0
  );
  const bonus = rawUpperSum >= UPPER_SCORE_CAP ? UPPER_BONUS : 0;
  const totalScore = computeTotalScore(categories, bonus);
  const scoredCategories = computeScoredMask(categories);
  const scoredCount = categories.filter((c) => c.isScored).length;
  return { upperScore, bonus, totalScore, scoredCategories, scoredCount };
}

export function computeAccumulatedScore(state: GameState): number {
  let sum = 0;
  for (const cat of state.categories) {
    if (cat.isScored) sum += cat.score;
  }
  return sum + state.bonus;
}

/** Pure reducer for all game state transitions. Handles dice rolling, rerolling,
 *  category scoring, eval response integration, and debug/manual overrides. */
export function gameReducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case 'ROLL': {
      if (state.turnPhase !== 'idle') return state;
      const dice = Array.from({ length: TOTAL_DICE }, () => ({
        value: randomDie(),
        held: true,
      }));
      return {
        ...state,
        dice,
        rerollsRemaining: 2,
        lastEvalResponse: null,
        sortMap: null,
        turnPhase: 'rolled',
        pendingTrajectoryEvent: 'roll',
      };
    }

    case 'TOGGLE_DIE': {
      const dice = state.dice.map((d, i) =>
        i === action.index ? { ...d, held: !d.held } : d
      );
      return { ...state, dice };
    }

    case 'REROLL': {
      if (state.rerollsRemaining <= 0) return state;
      // Compute decision label and delta before rerolling
      const keptValues = state.dice.filter(d => d.held).map(d => d.value).sort((a, b) => a - b);
      const rerollLabel = keptValues.length === TOTAL_DICE
        ? 'Kept all'
        : keptValues.length === 0
          ? 'Rerolled all'
          : `Kept ${keptValues.join(',')}`;

      let rerollDelta: number | null = null;
      if (state.lastEvalResponse?.mask_evs && state.sortMap) {
        // Compute player's reroll mask in sorted order
        let playerMask = 0;
        for (let i = 0; i < state.dice.length; i++) {
          if (!state.dice[i].held) playerMask |= 1 << i;
        }
        // Map to sorted order
        let sortedMask = 0;
        for (let si = 0; si < state.sortMap.length; si++) {
          if (playerMask & (1 << state.sortMap[si])) {
            sortedMask |= 1 << si;
          }
        }
        const playerMaskEv = state.lastEvalResponse.mask_evs[sortedMask] ?? null;
        const optimalMaskEv = state.lastEvalResponse.optimal_mask_ev ?? null;
        if (playerMaskEv !== null && optimalMaskEv !== null) {
          rerollDelta = playerMaskEv - optimalMaskEv;
        }
      }

      const dice = state.dice.map((d) =>
        d.held ? d : { value: randomDie(), held: true }
      );
      return {
        ...state,
        dice,
        rerollsRemaining: state.rerollsRemaining - 1,
        lastEvalResponse: null,
        sortMap: null,
        pendingTrajectoryEvent: 'reroll',
        pendingRerollLabel: rerollLabel,
        pendingRerollDelta: rerollDelta,
      };
    }

    case 'SCORE_CATEGORY': {
      if (state.turnPhase !== 'rolled') return state;
      const catId = action.categoryId;
      const cat = state.categories[catId];
      if (!cat || cat.isScored || !cat.available) return state;

      const categories = state.categories.map((c) =>
        c.id === catId
          ? { ...c, isScored: true, score: c.suggestedScore }
          : c.isScored ? c : { ...c, suggestedScore: 0, evIfScored: 0, available: true }
      );

      const derived = recomputeDerived(categories);
      const turnPhase = derived.scoredCount >= CATEGORY_COUNT ? 'game_over' as const : 'idle' as const;

      const evIfScored = state.categories[catId].evIfScored;
      const accScore = derived.totalScore;
      const trajectory = state.trajectory;
      const scoreLabel = `${cat.name} (${cat.suggestedScore})`;
      const optimalCatEv = state.lastEvalResponse?.optimal_category_ev ?? null;
      let scoreDelta: number | undefined;
      if (optimalCatEv !== null) {
        scoreDelta = evIfScored - optimalCatEv;
      }
      // The solver's ev_if_scored = category_score + V(successor_state).
      // V(successor) already accounts for the terminal bonus (50 pts if upper >= 63).
      // Use raw scored sum (no bonus) + V(successor) to avoid double-counting
      // both the category score (Bug 1) and the upper bonus (Bug 2).
      const successorEv = evIfScored - cat.suggestedScore;
      const rawScoredSum = categories.reduce((s, c) => c.isScored ? s + c.score : s, 0);
      const point: TrajectoryPoint = {
        index: trajectory.length,
        turn: derived.scoredCount,
        event: 'score',
        expectedFinal: turnPhase === 'game_over' ? accScore : rawScoredSum + successorEv,
        accumulatedScore: accScore,
        stateEv: turnPhase === 'game_over' ? 0 : successorEv,
        label: scoreLabel,
        delta: scoreDelta,
      };

      return {
        ...state,
        categories,
        upperScore: derived.upperScore,
        totalScore: derived.totalScore,
        bonus: derived.bonus,
        scoredCategories: derived.scoredCategories,
        lastEvalResponse: null,
        sortMap: null,
        turnPhase,
        rerollsRemaining: 0,
        trajectory: [...trajectory, point],
        pendingTrajectoryEvent: null,
      };
    }

    case 'SET_EVAL_RESPONSE': {
      const categories = state.categories.map((cat) => {
        const apiCat = action.response.categories.find((c) => c.id === cat.id);
        if (!apiCat) return cat;
        return {
          ...cat,
          suggestedScore: apiCat.score,
          evIfScored: apiCat.ev_if_scored,
          available: apiCat.available,
        };
      });
      let trajectory = state.trajectory;
      if (state.pendingTrajectoryEvent) {
        const accScore = computeAccumulatedScore(state);
        // Raw sum without bonus — the solver's state_ev already includes
        // the terminal bonus (50 pts) when upper >= 63.
        const rawScoredSum = state.categories.reduce((s, c) => c.isScored ? s + c.score : s, 0);
        const turn = state.categories.filter((c) => c.isScored).length;
        const point: TrajectoryPoint = {
          index: trajectory.length,
          turn,
          event: state.pendingTrajectoryEvent,
          expectedFinal: rawScoredSum + action.response.state_ev,
          accumulatedScore: accScore,
          stateEv: action.response.state_ev,
          label: state.pendingRerollLabel ?? undefined,
          delta: state.pendingRerollDelta ?? undefined,
        };
        trajectory = [...trajectory, point];
      }
      return {
        ...state,
        categories,
        lastEvalResponse: action.response,
        sortMap: action.sortMap,
        trajectory,
        pendingTrajectoryEvent: null,
        pendingRerollLabel: null,
        pendingRerollDelta: null,
      };
    }

    case 'TOGGLE_DEBUG':
      return { ...state, showDebug: !state.showDebug };

    case 'TOGGLE_HINTS':
      return { ...state, showHints: !state.showHints };

    case 'SET_DIE_VALUE': {
      const { index, value } = action;
      if (index < 0 || index >= TOTAL_DICE || value < 0 || value > 6) return state;
      let dice = state.dice.map((d, i) =>
        i === index ? { ...d, value } : d
      );
      let turnPhase = state.turnPhase;
      let rerollsRemaining = state.rerollsRemaining;
      // If idle or game_over, transition to rolled and fill zero dice with 1
      if (turnPhase === 'idle' || turnPhase === 'game_over') {
        turnPhase = 'rolled';
        rerollsRemaining = 2;
        dice = dice.map((d) => ({
          value: d.value === 0 ? 1 : d.value,
          held: true,
        }));
      }
      return {
        ...state,
        dice,
        turnPhase,
        rerollsRemaining,
        lastEvalResponse: null,
        sortMap: null,
      };
    }

    case 'SET_REROLLS': {
      if (state.turnPhase !== 'rolled') return state;
      const rerollsRemaining = Math.max(0, Math.min(2, action.rerollsRemaining));
      return {
        ...state,
        rerollsRemaining,
        lastEvalResponse: null,
        sortMap: null,
      };
    }

    case 'SET_CATEGORY_SCORE': {
      const { categoryId, score } = action;
      const cat = state.categories[categoryId];
      if (!cat) return state;

      const categories = state.categories.map((c) =>
        c.id === categoryId
          ? { ...c, isScored: true, score }
          : c
      );

      const derived = recomputeDerived(categories);
      const turnPhase = derived.scoredCount >= CATEGORY_COUNT ? 'game_over' as const : 'idle' as const;

      return {
        ...state,
        categories,
        upperScore: derived.upperScore,
        totalScore: derived.totalScore,
        bonus: derived.bonus,
        scoredCategories: derived.scoredCategories,
        lastEvalResponse: null,
        sortMap: null,
        turnPhase,
        rerollsRemaining: 0,
      };
    }

    case 'UNSET_CATEGORY': {
      const { categoryId } = action;
      const cat = state.categories[categoryId];
      if (!cat || !cat.isScored) return state;

      const categories = state.categories.map((c) =>
        c.id === categoryId
          ? { ...c, isScored: false, score: 0 }
          : c
      );

      const derived = recomputeDerived(categories);
      const turnPhase = state.turnPhase === 'game_over' ? 'idle' as const : state.turnPhase;

      return {
        ...state,
        categories,
        upperScore: derived.upperScore,
        totalScore: derived.totalScore,
        bonus: derived.bonus,
        scoredCategories: derived.scoredCategories,
        lastEvalResponse: null,
        sortMap: null,
        turnPhase,
      };
    }

    case 'RESET_GAME':
      clearSavedState();
      return freshState();

    case 'SET_INITIAL_EV': {
      if (state.trajectory.length > 0) return state;
      const point: TrajectoryPoint = {
        index: 0,
        turn: 0,
        event: 'start',
        expectedFinal: action.ev,
        accumulatedScore: 0,
        stateEv: action.ev,
      };
      return { ...state, trajectory: [point] };
    }

    case 'SET_DENSITY_RESULT': {
      const traj = state.trajectory.map((p) =>
        p.index === action.index ? { ...p, percentiles: action.percentiles } : p
      );
      return { ...state, trajectory: traj };
    }

    default:
      return state;
  }
}
