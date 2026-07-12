import type { GameState, GameStateSnapshot, GameAction, CategoryState, TrajectoryPoint } from './types.ts';
import { CATEGORY_NAMES, CATEGORY_COUNT, TOTAL_DICE, UPPER_SCORE_CAP, UPPER_BONUS, UPPER_CATEGORIES } from './constants.ts';
import { loadGame, loadPrefs, clearGame } from './persistence.ts';

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

/** Just the game-data portion, shared by New Game and the initial fresh state. */
function freshSnapshot(): GameStateSnapshot {
  return {
    dice: Array.from({ length: TOTAL_DICE }, () => ({ value: 0, kept: false })),
    upperScore: 0,
    scoredCategories: 0,
    rerollsRemaining: 0,
    categories: buildInitialCategories(),
    totalScore: 0,
    bonus: 0,
    rawScoredSum: 0,
    lastEvalResponse: null,
    turnPhase: 'idle',
    trajectory: [],
  };
}

export function initialState(): GameState {
  const savedGame = loadGame();
  const snapshot: GameStateSnapshot = savedGame
    ? { ...freshSnapshot(), ...savedGame, lastEvalResponse: null }
    : freshSnapshot();
  return { ...snapshot, undoStack: [], redoStack: [], prefs: loadPrefs() };
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
  const rawScoredSum = categories.reduce((s, c) => c.isScored ? s + c.score : s, 0);
  return { upperScore, bonus, totalScore, scoredCategories, scoredCount, rawScoredSum };
}

/** Extract a snapshot (excludes undo/redo stacks and prefs — undo must not
 *  change preferences). */
function takeSnapshot(state: GameState): GameStateSnapshot {
  const { undoStack: _, redoStack: __, prefs: ___, ...snap } = state;
  return snap;
}

/** Restore a snapshot into a full GameState, preserving history and prefs. */
function restoreSnapshot(
  snap: GameStateSnapshot,
  undoStack: GameStateSnapshot[],
  redoStack: GameStateSnapshot[],
  prefs: GameState['prefs'],
): GameState {
  return { ...snap, undoStack, redoStack, prefs };
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
        kept: true,
      }));
      return {
        ...state,
        dice,
        rerollsRemaining: 2,
        lastEvalResponse: null,
        turnPhase: 'rolled',
        redoStack: [],
      };
    }

    case 'TOGGLE_DIE': {
      const dice = state.dice.map((d, i) =>
        i === action.index ? { ...d, kept: !d.kept } : d
      );
      return { ...state, dice };
    }

    case 'REROLL': {
      if (state.rerollsRemaining <= 0) return state;
      const dice = state.dice.map((d) =>
        d.kept ? d : { value: randomDie(), kept: true }
      );
      return {
        ...state,
        dice,
        rerollsRemaining: state.rerollsRemaining - 1,
        lastEvalResponse: null,
      };
    }

    case 'SCORE_CATEGORY': {
      if (state.turnPhase !== 'rolled') return state;
      const catId = action.categoryId;
      const cat = state.categories[catId];
      if (!cat || cat.isScored || !cat.available) return state;

      // Snapshot current state for undo before transitioning
      const snapshot = takeSnapshot(state);

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
      // Use rawScoredSum (no bonus) + V(successor) to avoid double-counting.
      const successorEv = evIfScored - cat.suggestedScore;
      const point: TrajectoryPoint = {
        index: trajectory.length,
        turn: derived.scoredCount,
        event: 'score',
        expectedFinal: turnPhase === 'game_over' ? accScore : derived.rawScoredSum + successorEv,
        accumulatedScore: accScore,
        stateEv: turnPhase === 'game_over' ? 0 : successorEv,
        upperScore: derived.upperScore,
        scoredCategories: derived.scoredCategories,
        label: scoreLabel,
        delta: scoreDelta,
      };

      return {
        ...state,
        categories,
        upperScore: derived.upperScore,
        totalScore: derived.totalScore,
        bonus: derived.bonus,
        rawScoredSum: derived.rawScoredSum,
        scoredCategories: derived.scoredCategories,
        lastEvalResponse: null,
        turnPhase,
        rerollsRemaining: 0,
        trajectory: [...trajectory, point],
        undoStack: [...state.undoStack, snapshot],
        redoStack: [],
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
      if (action.trajectoryEvent) {
        const accScore = computeAccumulatedScore(state);
        const turn = state.categories.filter((c) => c.isScored).length;
        const point: TrajectoryPoint = {
          index: trajectory.length,
          turn,
          event: action.trajectoryEvent,
          expectedFinal: state.rawScoredSum + action.response.state_ev,
          accumulatedScore: accScore,
          stateEv: action.response.state_ev,
          label: action.rerollLabel,
          delta: action.rerollDelta,
        };
        trajectory = [...trajectory, point];
      }
      return {
        ...state,
        categories,
        lastEvalResponse: action.response,
        trajectory,
      };
    }

    case 'TOGGLE_DEBUG':
      return { ...state, prefs: { ...state.prefs, showDebug: !state.prefs.showDebug } };

    case 'TOGGLE_HINTS':
      return { ...state, prefs: { ...state.prefs, showHints: !state.prefs.showHints } };

    case 'TOGGLE_GUIDE':
      return { ...state, prefs: { ...state.prefs, guideOpen: !state.prefs.guideOpen } };

    case 'SET_AUTOPLAY_DELAY':
      return { ...state, prefs: { ...state.prefs, autoplayDelay: action.delay } };

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
          kept: true,
        }));
      }
      return {
        ...state,
        dice,
        turnPhase,
        rerollsRemaining,
        lastEvalResponse: null,
      };
    }

    case 'SET_REROLLS': {
      if (state.turnPhase !== 'rolled') return state;
      const rerollsRemaining = Math.max(0, Math.min(2, action.rerollsRemaining));
      return {
        ...state,
        rerollsRemaining,
        lastEvalResponse: null,
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
        rawScoredSum: derived.rawScoredSum,
        scoredCategories: derived.scoredCategories,
        lastEvalResponse: null,
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
        rawScoredSum: derived.rawScoredSum,
        scoredCategories: derived.scoredCategories,
        lastEvalResponse: null,
        turnPhase,
      };
    }

    case 'UNDO': {
      if (state.undoStack.length === 0) return state;
      const snap = state.undoStack[state.undoStack.length - 1];
      const newUndoStack = state.undoStack.slice(0, -1);
      return restoreSnapshot(snap, newUndoStack, [...state.redoStack, takeSnapshot(state)], state.prefs);
    }

    case 'REDO': {
      if (state.redoStack.length === 0) return state;
      const snap = state.redoStack[state.redoStack.length - 1];
      const newRedoStack = state.redoStack.slice(0, -1);
      return restoreSnapshot(snap, [...state.undoStack, takeSnapshot(state)], newRedoStack, state.prefs);
    }

    case 'RESET_GAME':
      // Clear the saved game but keep preferences (they live under their own key).
      clearGame();
      return { ...freshSnapshot(), undoStack: [], redoStack: [], prefs: state.prefs };

    case 'SET_INITIAL_EV': {
      if (state.trajectory.length > 0) return state;
      const point: TrajectoryPoint = {
        index: 0,
        turn: 0,
        event: 'start',
        expectedFinal: action.ev,
        accumulatedScore: 0,
        stateEv: action.ev,
        upperScore: 0,
        scoredCategories: 0,
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
