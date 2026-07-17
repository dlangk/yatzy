/**
 * Pure dice-mask helpers and the DiceBar view model.
 *
 * This module is the single seam where the dice bar's display semantics live, so
 * they can be tested in node without a DOM. DiceBar.ts renders this and nothing
 * else. Keep this file free of runtime imports (types only).
 *
 * Mask convention: bit i set means "reroll die i". Masks are in original dice
 * order and are never sorted.
 */
import type { DieState, EvaluateResponse, GameState } from './types.ts';

/** What the solver would do with a die. */
export type DieAdvice = 'HOLD' | 'REROLL';

export interface DieViewModel {
  /** Die face, 1-6. 0 renders the blank "?" placeholder. */
  value: number;
  held: boolean;
  /** Button text. Empty string before the first roll of a turn. */
  label: DieAdvice | '';
  disabled: boolean;
  faded: boolean;
}

export interface DiceBarViewModel {
  dice: DieViewModel[];
  /** Solver advice per die, or null when the advice row is hidden. */
  advice: DieAdvice[] | null;
  /** Per-die mismatch markers. All false unless the player's mask is costly. */
  ticks: boolean[];
  /** Expected points given up by the player's current mask, or null when there is
   *  nothing to report. Always negative when present. */
  cost: number | null;
}

/**
 * Half the displayed precision. A mask whose cost is inside this band is treated
 * as free: two distinct masks can tie on EV, and a costless tie must not be
 * flagged red, because red means "giving up points".
 */
export const COST_EPS = 0.005;

/** Build the reroll mask for the player's current selection. Bit i set = reroll die i. */
export function rerollMaskFromDice(dice: readonly DieState[]): number {
  let mask = 0;
  for (let i = 0; i < dice.length; i++) {
    if (!dice[i].kept) mask |= 1 << i;
  }
  return mask;
}

/**
 * Expected points given up by `mask` versus the solver's best mask.
 * Returns 0 for the optimal mask, a negative number for a worse one, and null
 * when the response lacks the data to say.
 */
export function maskCostVsOptimal(ev: EvaluateResponse | null, mask: number): number | null {
  const playerEv = ev?.mask_evs?.[mask];
  const optimalEv = ev?.optimal_mask_ev;
  if (playerEv === undefined || optimalEv === undefined) return null;
  return playerEv - optimalEv;
}

/** Project game state into everything the dice bar needs to render. */
export function diceBarViewModel(s: GameState): DiceBarViewModel {
  const active = s.turnPhase === 'rolled';
  const hasEval = s.lastEvalResponse !== null;
  const canToggle = active && hasEval && s.rerollsRemaining > 0;
  const optimalMask = s.lastEvalResponse?.optimal_mask ?? null;

  const dice: DieViewModel[] = s.dice.map(d => ({
    value: active ? d.value : 0,
    held: active ? d.kept : true,
    label: active ? (d.kept ? 'HOLD' : 'REROLL') : '',
    disabled: !canToggle,
    faded: !active,
  }));

  const noTicks = s.dice.map(() => false);

  const showAdvice = s.prefs.showHints && active && hasEval
    && optimalMask !== null && s.rerollsRemaining > 0;
  if (!showAdvice) {
    return { dice, advice: null, ticks: noTicks, cost: null };
  }

  const advice: DieAdvice[] = s.dice.map(
    (_, i) => ((optimalMask! & (1 << i)) ? 'REROLL' : 'HOLD'),
  );

  const rawCost = maskCostVsOptimal(s.lastEvalResponse, rerollMaskFromDice(s.dice));
  const costly = rawCost !== null && rawCost < -COST_EPS;
  if (!costly) {
    return { dice, advice, ticks: noTicks, cost: null };
  }

  const ticks = s.dice.map((d, i) => d.kept !== (advice[i] === 'HOLD'));
  return { dice, advice, ticks, cost: rawCost };
}
