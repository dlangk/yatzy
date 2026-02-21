import { describe, it, expect, vi, beforeEach } from 'vitest';
import { gameReducer } from './reducer.js';
import { CATEGORY_NAMES, CATEGORY_COUNT, TOTAL_DICE, BONUS_THRESHOLD, BONUS_SCORE } from './constants.js';

const storageMock = { getItem: vi.fn(), setItem: vi.fn(), removeItem: vi.fn() };
vi.stubGlobal('localStorage', storageMock);

function buildCategories() {
  return CATEGORY_NAMES.map((name, id) => ({
    id, name, score: 0, isScored: false,
    suggestedScore: 0, evIfScored: 0, available: true,
  }));
}

function freshState() {
  return {
    dice: Array.from({ length: TOTAL_DICE }, () => ({ value: 0, held: false })),
    upperScore: 0,
    scoredCategories: 0,
    rerollsRemaining: 0,
    categories: buildCategories(),
    totalScore: 0,
    bonus: 0,
    lastEvalResponse: null,
    sortMap: null,
    showDebug: false,
    turnPhase: 'idle',
    history: [],
  };
}

function rolledState() {
  return {
    ...freshState(),
    dice: [
      { value: 3, held: true },
      { value: 1, held: true },
      { value: 4, held: true },
      { value: 1, held: true },
      { value: 5, held: true },
    ],
    rerollsRemaining: 2,
    turnPhase: 'rolled',
  };
}

describe('gameReducer', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  describe('ROLL', () => {
    it('produces 5 dice with values 1-6, sets rerollsRemaining=2, phase=rolled', () => {
      const state = freshState();
      const next = gameReducer(state, { type: 'ROLL' });
      expect(next.dice).toHaveLength(TOTAL_DICE);
      for (const d of next.dice) {
        expect(d.value).toBeGreaterThanOrEqual(1);
        expect(d.value).toBeLessThanOrEqual(6);
        expect(d.held).toBe(true);
      }
      expect(next.rerollsRemaining).toBe(2);
      expect(next.turnPhase).toBe('rolled');
      expect(next.lastEvalResponse).toBeNull();
      expect(next.sortMap).toBeNull();
    });

    it('is ignored when phase !== idle', () => {
      const state = rolledState();
      const next = gameReducer(state, { type: 'ROLL' });
      expect(next).toBe(state);
    });
  });

  describe('REROLL', () => {
    it('rerolls unheld dice and decrements rerollsRemaining', () => {
      const state = {
        ...rolledState(),
        dice: [
          { value: 3, held: true },
          { value: 1, held: false },
          { value: 4, held: true },
          { value: 1, held: false },
          { value: 5, held: true },
        ],
      };
      const next = gameReducer(state, { type: 'REROLL' });
      expect(next.dice[0].value).toBe(3);
      expect(next.dice[2].value).toBe(4);
      expect(next.dice[4].value).toBe(5);
      for (const d of next.dice) {
        expect(d.held).toBe(true);
      }
      expect(next.rerollsRemaining).toBe(1);
      expect(next.lastEvalResponse).toBeNull();
    });

    it('is ignored when rerollsRemaining <= 0', () => {
      const state = { ...rolledState(), rerollsRemaining: 0 };
      const next = gameReducer(state, { type: 'REROLL' });
      expect(next).toBe(state);
    });
  });

  describe('SET_EVAL_RESPONSE', () => {
    it('maps all API fields into category state including score=0', () => {
      const state = rolledState();
      const response = {
        categories: [
          { id: 0, name: 'Ones', score: 2, available: true, ev_if_scored: 180.5 },
          { id: 6, name: 'One Pair', score: 0, available: true, ev_if_scored: 170.0 },
          { id: 14, name: 'Yatzy', score: 0, available: true, ev_if_scored: 165.0 },
        ],
        optimal_category: 0,
        optimal_category_ev: 180.5,
        state_ev: 200.0,
      };
      const sortMap = [1, 3, 0, 2, 4];
      const next = gameReducer(state, { type: 'SET_EVAL_RESPONSE', response, sortMap });

      expect(next.categories[0].suggestedScore).toBe(2);
      expect(next.categories[0].evIfScored).toBe(180.5);
      expect(next.categories[0].available).toBe(true);
      expect(next.categories[6].suggestedScore).toBe(0);
      expect(next.categories[6].evIfScored).toBe(170.0);
      expect(next.categories[14].suggestedScore).toBe(0);
      expect(next.lastEvalResponse).toBe(response);
      expect(next.sortMap).toBe(sortMap);
    });
  });

  describe('SCORE_CATEGORY', () => {
    it('commits suggestedScore as the final score', () => {
      const state = rolledState();
      state.categories[0].suggestedScore = 3;
      state.categories[0].available = true;
      const next = gameReducer(state, { type: 'SCORE_CATEGORY', categoryId: 0 });
      expect(next.categories[0].isScored).toBe(true);
      expect(next.categories[0].score).toBe(3);
      expect(next.turnPhase).toBe('idle');
      expect(next.rerollsRemaining).toBe(0);
    });

    it('commits score=0 correctly (regression test for falsy-zero bug)', () => {
      const state = rolledState();
      state.categories[14].suggestedScore = 0;
      state.categories[14].available = true;
      const next = gameReducer(state, { type: 'SCORE_CATEGORY', categoryId: 14 });
      expect(next.categories[14].isScored).toBe(true);
      expect(next.categories[14].score).toBe(0);
    });

    it('is ignored when phase !== rolled', () => {
      const state = freshState();
      state.categories[0].suggestedScore = 5;
      const next = gameReducer(state, { type: 'SCORE_CATEGORY', categoryId: 0 });
      expect(next).toBe(state);
    });

    it('is ignored for already-scored categories', () => {
      const state = rolledState();
      state.categories[0].isScored = true;
      const next = gameReducer(state, { type: 'SCORE_CATEGORY', categoryId: 0 });
      expect(next).toBe(state);
    });
  });

  describe('upper score and bonus', () => {
    it('computes upper score from first 6 categories', () => {
      const state = rolledState();
      state.categories[0].suggestedScore = 3;
      state.categories[0].available = true;
      let next = gameReducer(state, { type: 'SCORE_CATEGORY', categoryId: 0 });
      expect(next.upperScore).toBe(3);

      next = { ...next, turnPhase: 'rolled' };
      next.categories[1].suggestedScore = 6;
      next.categories[1].available = true;
      next = gameReducer(next, { type: 'SCORE_CATEGORY', categoryId: 1 });
      expect(next.upperScore).toBe(9);
      expect(next.bonus).toBe(0);
    });

    it('awards 50-point bonus when upper score >= 63', () => {
      const state = rolledState();
      const upperScores = [3, 6, 9, 12, 15];
      for (let i = 0; i < 5; i++) {
        state.categories[i].isScored = true;
        state.categories[i].score = upperScores[i];
      }
      state.categories[5].suggestedScore = 18;
      state.categories[5].available = true;
      const next = gameReducer(state, { type: 'SCORE_CATEGORY', categoryId: 5 });
      expect(next.upperScore).toBe(BONUS_THRESHOLD);
      expect(next.bonus).toBe(BONUS_SCORE);
      expect(next.totalScore).toBe(3 + 6 + 9 + 12 + 15 + 18 + BONUS_SCORE);
    });
  });

  describe('game over', () => {
    it('transitions to game_over when all 15 categories are scored', () => {
      const state = rolledState();
      for (let i = 0; i < CATEGORY_COUNT - 1; i++) {
        state.categories[i].isScored = true;
        state.categories[i].score = 1;
      }
      const lastId = CATEGORY_COUNT - 1;
      state.categories[lastId].suggestedScore = 50;
      state.categories[lastId].available = true;
      const next = gameReducer(state, { type: 'SCORE_CATEGORY', categoryId: lastId });
      expect(next.turnPhase).toBe('game_over');
    });
  });

  describe('RESET_GAME', () => {
    it('returns to fresh state', () => {
      const state = rolledState();
      state.categories[0].isScored = true;
      state.categories[0].score = 5;
      state.totalScore = 5;
      const next = gameReducer(state, { type: 'RESET_GAME' });
      expect(next.turnPhase).toBe('idle');
      expect(next.totalScore).toBe(0);
      expect(next.dice.every((d) => d.value === 0)).toBe(true);
      expect(next.categories.every((c) => !c.isScored)).toBe(true);
      expect(next.rerollsRemaining).toBe(0);
    });
  });

  describe('PUSH_HISTORY', () => {
    it('appends entry to history array', () => {
      const state = freshState();
      const entry = { type: 'start', turn: 0, label: null, delta: null, stateEv: 200, expectedFinal: 200, accumulatedScore: 0, density: null };
      const next = gameReducer(state, { type: 'PUSH_HISTORY', entry });
      expect(next.history).toHaveLength(1);
      expect(next.history[0]).toEqual(entry);
    });
  });

  describe('PATCH_HISTORY', () => {
    it('patches existing entry at given index', () => {
      const state = { ...freshState(), history: [
        { type: 'reroll', turn: 1, stateEv: null, expectedFinal: null },
      ]};
      const next = gameReducer(state, { type: 'PATCH_HISTORY', index: 0, patch: { stateEv: 180, expectedFinal: 200 } });
      expect(next.history[0].stateEv).toBe(180);
      expect(next.history[0].expectedFinal).toBe(200);
      expect(next.history[0].type).toBe('reroll');
    });
  });

  describe('TOGGLE_DIE', () => {
    it('toggles held state of targeted die', () => {
      const state = rolledState();
      expect(state.dice[2].held).toBe(true);
      const next = gameReducer(state, { type: 'TOGGLE_DIE', index: 2 });
      expect(next.dice[2].held).toBe(false);
      expect(next.dice[0].held).toBe(true);
    });
  });
});
