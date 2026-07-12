/**
 * Plain-language explanations for the analytical terms in the Play UI.
 * Centralized so the wording is easy to tune. Consumed via attachTooltip().
 */
export const TIP = {
  // EvalPanel — This Turn
  bestKeep: 'The dice the optimal strategy would hold right now.',
  keepEv:
    'Expected final score if you hold the best dice, then play on optimally. ' +
    'EV means expected value: the long-run average.',
  yourKeepEv:
    'Expected final score given the dice you are currently holding, then playing on optimally. ' +
    'Compare it with Keep EV to see what your choice costs.',
  bestScore: 'The category the optimal strategy would use for this roll, and the points it scores.',
  scoreEv:
    'Expected final score if you score this roll now in the best category and play the rest optimally.',

  // EvalPanel — Game
  expectedFinal:
    'Your projected final score: points banked so far plus the optimal expected value of the turns you have left.',
  finishRange:
    'Where your final score will most likely land. Eight games in ten finish inside this range ' +
    '(the 10th to 90th percentile).',
  currentScore: 'Points banked so far, including the 50-point upper bonus once you have earned it.',
  optimalPercentile:
    'How your projected final ranks against a perfect player’s scores. ' +
    'p50 is the median; p90 beats 90% of optimal games.',
  turn: 'How many of the 15 categories you have filled.',

  // Dice legend
  legendKept: 'A die you are holding. It will not be rerolled.',
  legendReroll: 'A die that will be rerolled.',
  legendOptimal: 'Green outline: the optimal play is to hold this die.',
  legendSuboptimal: 'Red outline: the optimal play is to reroll this die; holding it gives up points.',

  // Scorecard
  scEfinal:
    'Expected final score if you score this roll in that category now and play on optimally. ' +
    'The highlighted row is the best choice.',

  // Decision Log
  logDelta:
    'How much each decision cost versus the optimal move, in expected final points. ' +
    '0 means you matched optimal play; negative means you gave up points.',
  logEfinal: 'Your projected final score after that decision.',
  logTotalDelta: 'Total expected points given up across the game versus perfect play.',

  // Dice (composed dynamically from the current state, see DiceBar)
  dieKept: 'You are holding this die.',
  dieReroll: 'This die will be rerolled.',
  dieAgree: ' Optimal play agrees.',
  dieKeepButReroll: 'You are holding this die, but optimal play is to reroll it.',
  dieRerollButKeep: 'This die will be rerolled, but optimal play is to hold it.',
} as const;
