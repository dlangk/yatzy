import { describe, it, expect } from 'vitest';
import { rerollMaskFromDice, maskCostVsOptimal, diceBarViewModel, COST_EPS } from './mask.ts';
import type { GameState, DieState, EvaluateResponse } from './types.ts';

function dice(spec: [number, boolean][]): DieState[] {
  return spec.map(([value, kept]) => ({ value, kept }));
}

/** mask_evs has 32 entries (one per 5-bit reroll mask). Optimal mask is 0b00110 (reroll dice 1 and 2). */
function evalResponse(over: Partial<EvaluateResponse> = {}): EvaluateResponse {
  const mask_evs = Array.from({ length: 32 }, () => 100);
  mask_evs[0b00110] = 110; // the optimal mask scores best
  return {
    mask_evs,
    optimal_mask: 0b00110,
    optimal_mask_ev: 110,
    categories: [],
    optimal_category: 0,
    optimal_category_ev: 50,
    state_ev: 100,
    ...over,
  };
}

function state(over: Partial<GameState> = {}): GameState {
  return {
    dice: dice([[1, true], [2, false], [3, false], [4, true], [5, true]]),
    upperScore: 0,
    scoredCategories: 0,
    rerollsRemaining: 2,
    categories: [],
    totalScore: 0,
    bonus: 0,
    rawScoredSum: 0,
    lastEvalResponse: evalResponse(),
    detents: null,
    turnPhase: 'rolled',
    trajectory: [],
    undoStack: [],
    redoStack: [],
    prefs: { showHints: true, showDebug: false, guideOpen: false, autoplayDelay: 300, override: false, riskLevel: 2 },
    ...over,
  } as GameState;
}

describe('rerollMaskFromDice', () => {
  it('returns 0 when every die is held', () => {
    expect(rerollMaskFromDice(dice([[1, true], [2, true], [3, true], [4, true], [5, true]]))).toBe(0);
  });

  it('returns 31 when no die is held', () => {
    expect(rerollMaskFromDice(dice([[1, false], [2, false], [3, false], [4, false], [5, false]]))).toBe(31);
  });

  it('sets bit i for each die that will be rerolled, in original dice order', () => {
    // dice 1 and 2 are not held -> 0b00110 == 6
    expect(rerollMaskFromDice(dice([[1, true], [2, false], [3, false], [4, true], [5, true]]))).toBe(0b00110);
  });
});

describe('maskCostVsOptimal', () => {
  it('returns 0 when the player mask is the optimal mask', () => {
    expect(maskCostVsOptimal(evalResponse(), 0b00110)).toBe(0);
  });

  it('returns a negative number for a worse mask', () => {
    expect(maskCostVsOptimal(evalResponse(), 0b00000)).toBe(-10);
  });

  it('returns null when mask_evs is absent', () => {
    expect(maskCostVsOptimal(evalResponse({ mask_evs: undefined }), 0)).toBeNull();
  });

  it('returns null when optimal_mask_ev is absent', () => {
    expect(maskCostVsOptimal(evalResponse({ optimal_mask_ev: undefined }), 0)).toBeNull();
  });

  it('returns null when the eval response is null', () => {
    expect(maskCostVsOptimal(null, 0)).toBeNull();
  });
});

describe('diceBarViewModel advice', () => {
  it('says REROLL where optimal_mask has the bit set and HOLD where it does not', () => {
    // optimal_mask 0b00110 -> reroll dice 1 and 2
    expect(diceBarViewModel(state()).advice)
      .toEqual(['HOLD', 'REROLL', 'REROLL', 'HOLD', 'HOLD']);
  });

  // REGRESSION for the shipped bug: the legend said green meant "hold" while the code
  // made it mean "you agree with the solver", so advice moved when the player toggled.
  it('is identical for two states differing only in kept flags', () => {
    const allHeld = state({ dice: dice([[1, true], [2, true], [3, true], [4, true], [5, true]]) });
    const noneHeld = state({ dice: dice([[1, false], [2, false], [3, false], [4, false], [5, false]]) });
    expect(diceBarViewModel(allHeld).advice).toEqual(diceBarViewModel(noneHeld).advice);
  });

  it('does not change when a single die is toggled', () => {
    const before = diceBarViewModel(state()).advice;
    const toggled = state({ dice: dice([[1, false], [2, false], [3, false], [4, true], [5, true]]) });
    expect(diceBarViewModel(toggled).advice).toEqual(before);
  });
});

describe('diceBarViewModel ticks and cost', () => {
  it('shows no ticks and no cost when the player matches the optimal mask', () => {
    const vm = diceBarViewModel(state());
    expect(vm.ticks).toEqual([false, false, false, false, false]);
    expect(vm.cost).toBeNull();
  });

  // A costless tie must never be flagged red. Red means "giving up points".
  it('shows no ticks for a different mask that ties the optimal mask on EV', () => {
    const evs = Array.from({ length: 32 }, () => 0);
    evs[0b00110] = 110;
    evs[0b11000] = 110; // different mask, identical EV
    const vm = diceBarViewModel(state({
      dice: dice([[1, true], [2, true], [3, true], [4, false], [5, false]]), // mask 0b11000
      lastEvalResponse: evalResponse({ mask_evs: evs }),
    }));
    expect(vm.ticks).toEqual([false, false, false, false, false]);
    expect(vm.cost).toBeNull();
  });

  it('ticks exactly the dice whose choice differs from the advice when the mask is costly', () => {
    // Player holds everything (mask 0), optimal is 0b00110, cost -10.
    const vm = diceBarViewModel(state({
      dice: dice([[1, true], [2, true], [3, true], [4, true], [5, true]]),
    }));
    expect(vm.advice).toEqual(['HOLD', 'REROLL', 'REROLL', 'HOLD', 'HOLD']);
    expect(vm.ticks).toEqual([false, true, true, false, false]);
    expect(vm.cost).toBe(-10);
  });

  it('does not tick when the cost is inside EPS', () => {
    const evs = Array.from({ length: 32 }, () => 0);
    evs[0b00110] = 110;
    evs[0b00000] = 110 - (COST_EPS / 2); // -0.0025, below the threshold
    const vm = diceBarViewModel(state({
      dice: dice([[1, true], [2, true], [3, true], [4, true], [5, true]]),
      lastEvalResponse: evalResponse({ mask_evs: evs }),
    }));
    expect(vm.ticks).toEqual([false, false, false, false, false]);
    expect(vm.cost).toBeNull();
  });

  it('ticks when the cost is just past EPS', () => {
    const evs = Array.from({ length: 32 }, () => 0);
    evs[0b00110] = 110;
    evs[0b00000] = 110 - (COST_EPS * 2); // -0.01, past the threshold
    const vm = diceBarViewModel(state({
      dice: dice([[1, true], [2, true], [3, true], [4, true], [5, true]]),
      lastEvalResponse: evalResponse({ mask_evs: evs }),
    }));
    expect(vm.ticks).toEqual([false, true, true, false, false]);
    expect(vm.cost).toBeCloseTo(-0.01, 6);
  });
});

describe('diceBarViewModel hidden states', () => {
  it('hides advice when hints are off but still populates the dice', () => {
    const vm = diceBarViewModel(state({
      prefs: { showHints: false, showDebug: false, guideOpen: false, autoplayDelay: 300, override: false, riskLevel: 2 },
    }));
    expect(vm.advice).toBeNull();
    expect(vm.cost).toBeNull();
    expect(vm.ticks).toEqual([false, false, false, false, false]);
    expect(vm.dice).toHaveLength(5);
  });

  it('hides advice when no rerolls remain', () => {
    expect(diceBarViewModel(state({ rerollsRemaining: 0 })).advice).toBeNull();
  });

  it('hides advice when the turn is not in the rolled phase', () => {
    expect(diceBarViewModel(state({ turnPhase: 'idle' })).advice).toBeNull();
  });

  it('hides advice when there is no eval response', () => {
    expect(diceBarViewModel(state({ lastEvalResponse: null })).advice).toBeNull();
  });
});

describe('diceBarViewModel dice', () => {
  it('labels held dice HOLD and unheld dice REROLL', () => {
    expect(diceBarViewModel(state()).dice.map(d => d.label))
      .toEqual(['HOLD', 'REROLL', 'REROLL', 'HOLD', 'HOLD']);
  });

  it('blanks values and labels and fades every die when the turn is not rolled', () => {
    const vm = diceBarViewModel(state({ turnPhase: 'idle' }));
    expect(vm.dice.map(d => d.value)).toEqual([0, 0, 0, 0, 0]);
    expect(vm.dice.map(d => d.label)).toEqual(['', '', '', '', '']);
    expect(vm.dice.every(d => d.faded)).toBe(true);
    expect(vm.dice.every(d => d.disabled)).toBe(true);
  });

  it('disables the dice when no rerolls remain but keeps them unfaded', () => {
    const vm = diceBarViewModel(state({ rerollsRemaining: 0 }));
    expect(vm.dice.every(d => d.disabled)).toBe(true);
    expect(vm.dice.every(d => d.faded)).toBe(false);
  });
});
