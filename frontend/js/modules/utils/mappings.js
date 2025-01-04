import gameState from "../game/gameState.js";

export function createMappingForSortedDice(diceValues) {
    // Pair dice values with indices, sort by value, and generate mapping
    const pairedDice = diceValues.map((value, index) => ({ value, index }));
    const sortedDice = [...pairedDice].sort((a, b) => a.value - b.value);

    const mapping = sortedDice.map((pair, sortedIndex) => ({
        sortedIndex,
        originalIndex: pair.index,
    }));

    return { sortedDice: sortedDice.map(pair => pair.value), mapping };
}

export function applyMappingToRerollMask(mask, mapping) {
    const reorderedMask = new Array(mask.length);
    mapping.forEach(({ sortedIndex, originalIndex }) => {
        reorderedMask[sortedIndex] = mask[originalIndex];
    });
    return reorderedMask.join("");
}

export function reverseMapping(mask, mapping) {
    const originalMask = new Array(mask.length);
    mapping.forEach(({ sortedIndex, originalIndex }) => {
        originalMask[originalIndex] = mask[sortedIndex];
    });
    return originalMask.join("");
}

export function convertMaskToId(mask) {
    return mask.split("").reduce((id, bit, i) => (bit === "1" ? id | (1 << i) : id), 0);
}

export function getSortedDiceWithHighlights() {
    const activePlayer = gameState.getActivePlayer();
    if (!activePlayer) return null;

    // Retrieve dice values and reroll mask
    const diceValues = activePlayer.diceState.map(die => die.value);
    const rerollMask = activePlayer.optimalRerollMask || [];

    // Generate sorted dice with reroll highlights
    return diceValues.map((value, index) => ({value, index}))
        .sort((a, b) => a.value - b.value)
        .map(({value, index}) => ({
            value,
            isHighlighted: rerollMask[index] === "1",
        }));
}