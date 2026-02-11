import { describe, it, expect, vi, beforeEach } from 'vitest';
import { gameReducer } from './reducer.ts';
import type { GameState, CategoryState, EvaluateResponse } from './types.ts';
import { CATEGORY_NAMES, CATEGORY_COUNT, TOTAL_DICE, BONUS_THRESHOLD, BONUS_SCORE } from './constants.ts';

// Stub localStorage so the reducer never touches real storage
const storageMock = { getItem: vi.fn(), setItem: vi.fn(), removeItem: vi.fn() };
vi.stubGlobal('localStorage', storageMock);

function buildCategories(): CategoryState[] {
  return CATEGORY_NAMES.map((name, id) => ({
    id, name, score: 0, isScored: false,
    suggestedScore: 0, evIfScored: 0, available: true,
  }));
}

function freshState(): GameState {
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
  };
}

function rolledState(): GameState {
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

  // --- ROLL ---
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
      expect(next).toBe(state); // exact same reference
    });
  });

  // --- REROLL ---
  describe('REROLL', () => {
    it('rerolls unheld dice and decrements rerollsRemaining', () => {
      const state: GameState = {
        ...rolledState(),
        dice: [
          { value: 3, held: true },
          { value: 1, held: false },  // should change
          { value: 4, held: true },
          { value: 1, held: false },  // should change
          { value: 5, held: true },
        ],
      };
      const next = gameReducer(state, { type: 'REROLL' });
      // Held dice keep their values
      expect(next.dice[0].value).toBe(3);
      expect(next.dice[2].value).toBe(4);
      expect(next.dice[4].value).toBe(5);
      // All dice are held after reroll
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

  // --- SET_EVAL_RESPONSE ---
  describe('SET_EVAL_RESPONSE', () => {
    it('maps all API fields into category state including score=0', () => {
      const state = rolledState();
      const response: EvaluateResponse = {
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

      // Category 0 (Ones): score=2
      expect(next.categories[0].suggestedScore).toBe(2);
      expect(next.categories[0].evIfScored).toBe(180.5);
      expect(next.categories[0].available).toBe(true);

      // Category 6 (One Pair): score=0 — must be 0, not undefined
      expect(next.categories[6].suggestedScore).toBe(0);
      expect(next.categories[6].evIfScored).toBe(170.0);

      // Category 14 (Yatzy): score=0
      expect(next.categories[14].suggestedScore).toBe(0);

      expect(next.lastEvalResponse).toBe(response);
      expect(next.sortMap).toBe(sortMap);
    });
  });

  // --- SCORE_CATEGORY ---
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
      // Simulate: eval says Yatzy scores 0 but is optimal
      state.categories[14].suggestedScore = 0;
      state.categories[14].available = true;
      const next = gameReducer(state, { type: 'SCORE_CATEGORY', categoryId: 14 });
      expect(next.categories[14].isScored).toBe(true);
      expect(next.categories[14].score).toBe(0);
    });

    it('is ignored when phase !== rolled', () => {
      const state = freshState(); // phase=idle
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

  // --- Upper score and bonus ---
  describe('upper score and bonus', () => {
    it('computes upper score from first 6 categories', () => {
      const state = rolledState();
      // Score Ones=3, Twos=6 via two turns
      state.categories[0].suggestedScore = 3;
      state.categories[0].available = true;
      let next = gameReducer(state, { type: 'SCORE_CATEGORY', categoryId: 0 });
      expect(next.upperScore).toBe(3);

      // Score Twos in second turn
      next = { ...next, turnPhase: 'rolled' as const };
      next.categories[1].suggestedScore = 6;
      next.categories[1].available = true;
      next = gameReducer(next, { type: 'SCORE_CATEGORY', categoryId: 1 });
      expect(next.upperScore).toBe(9);
      expect(next.bonus).toBe(0); // not yet at threshold
    });

    it('awards 50-point bonus when upper score >= 63', () => {
      const state = rolledState();
      // Pre-score first 5 upper categories to get to 60
      const upperScores = [3, 6, 9, 12, 15]; // Ones=3, Twos=6, ..., Fives=15
      for (let i = 0; i < 5; i++) {
        state.categories[i].isScored = true;
        state.categories[i].score = upperScores[i];
      }
      // Score Sixes=18 to cross threshold (total=63)
      state.categories[5].suggestedScore = 18;
      state.categories[5].available = true;
      const next = gameReducer(state, { type: 'SCORE_CATEGORY', categoryId: 5 });
      expect(next.upperScore).toBe(BONUS_THRESHOLD);
      expect(next.bonus).toBe(BONUS_SCORE);
      expect(next.totalScore).toBe(3 + 6 + 9 + 12 + 15 + 18 + BONUS_SCORE);
    });
  });

  // --- Game over ---
  describe('game over', () => {
    it('transitions to game_over when all 15 categories are scored', () => {
      const state = rolledState();
      // Pre-score 14 categories
      for (let i = 0; i < CATEGORY_COUNT - 1; i++) {
        state.categories[i].isScored = true;
        state.categories[i].score = 1;
      }
      // Score the last one
      const lastId = CATEGORY_COUNT - 1;
      state.categories[lastId].suggestedScore = 50;
      state.categories[lastId].available = true;
      const next = gameReducer(state, { type: 'SCORE_CATEGORY', categoryId: lastId });
      expect(next.turnPhase).toBe('game_over');
    });
  });

  // --- RESET_GAME ---
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

  // --- TOGGLE_DIE ---
  describe('TOGGLE_DIE', () => {
    it('toggles held state of targeted die', () => {
      const state = rolledState();
      expect(state.dice[2].held).toBe(true);
      const next = gameReducer(state, { type: 'TOGGLE_DIE', index: 2 });
      expect(next.dice[2].held).toBe(false);
      // Other dice unchanged
      expect(next.dice[0].held).toBe(true);
    });
  });
});
