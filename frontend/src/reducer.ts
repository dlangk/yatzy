import type { GameState, GameAction, CategoryState } from './types.ts';
import { CATEGORY_NAMES, CATEGORY_COUNT, TOTAL_DICE, BONUS_THRESHOLD, BONUS_SCORE, UPPER_CATEGORIES } from './constants.ts';

const STORAGE_KEY = 'yatzy-game-state';

export function saveState(state: GameState): void {
  try {
    const { lastEvalResponse: _, sortMap: _s, ...persistable } = state;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(persistable));
  } catch { /* quota exceeded or private browsing — ignore */ }
}

function loadState(): GameState | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    // Restore transient fields
    return { ...parsed, lastEvalResponse: null, sortMap: null };
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
  return Math.min(sum, BONUS_THRESHOLD);
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
  const bonus = rawUpperSum >= BONUS_THRESHOLD ? BONUS_SCORE : 0;
  const totalScore = computeTotalScore(categories, bonus);
  const scoredCategories = computeScoredMask(categories);
  const scoredCount = categories.filter((c) => c.isScored).length;
  return { upperScore, bonus, totalScore, scoredCategories, scoredCount };
}

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
      const dice = state.dice.map((d) =>
        d.held ? d : { value: randomDie(), held: true }
      );
      return {
        ...state,
        dice,
        rerollsRemaining: state.rerollsRemaining - 1,
        lastEvalResponse: null,
        sortMap: null,
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
      return {
        ...state,
        categories,
        lastEvalResponse: action.response,
        sortMap: action.sortMap,
      };
    }

    case 'TOGGLE_DEBUG':
      return { ...state, showDebug: !state.showDebug };

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

    default:
      return state;
  }
}
