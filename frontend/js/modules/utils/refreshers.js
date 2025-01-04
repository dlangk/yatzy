import gameState from "../game/gameState.js";
import {API_ENDPOINTS, postJsonRequest} from "./endpoints.js";
import {applyMappingToRerollMask, convertMaskToId, createMappingForSortedDice, reverseMapping} from "./mappings.js";
import {renderUI} from "./uiBuilders.js";

const refreshers = [
    refreshAvailableCategories,
    refreshOptimalAction,
    refreshUserAction,
    refreshUserActionResult,
    refreshHistogram,
];

export async function runRefreshers() {
    console.log("Running refreshers...");
    try {
        for (const refresher of refreshers) {
            await refresher(); // Run each refresher sequentially
        }
        // Perform a single save operation after all refreshers
        gameState.saveState();
        renderUI(); // Perform a single render operation
    } catch (error) {
        console.error("Error running refreshers:", error);
    }
}

export async function refreshAvailableCategories() {
    const activePlayer = gameState.getActivePlayer();
    if (!activePlayer) {
        console.warn("No active player found.");
        return;
    }

    try {
        const payload = gameState.getEvaluationPayload();
        const response = await postJsonRequest(API_ENDPOINTS.AVAILABLE_CATEGORIES, payload);

        if (!response || !response.categories) {
            console.warn("No categories data in the response.");
            return;
        }

        // Update suggested scores and validity for each category
        const scorecard = gameState.getActivePlayerScorecard();
        response.categories.forEach(category => {
            const matchingCategory = scorecard.find(cat => cat.id === category.id);
            if (matchingCategory) {
                matchingCategory.isValid = category.valid;
                matchingCategory.suggestedScore = category.score; // Update suggested score
            }
        });
    } catch (error) {
        console.error("Error refreshing available categories:", error);
    }
}

export async function refreshOptimalAction() {
    console.log("Refreshing optimal action...");

    const activePlayer = gameState.getActivePlayer();
    if (!activePlayer) {
        console.warn("No active player found.");
        return;
    }

    const currentData = gameState.getEvaluationPayload();
    if (!currentData) {
        console.warn("Failed to generate evaluation payload.");
        return;
    }

    const { sortedDice, mapping } = createMappingForSortedDice(currentData.dice);
    currentData.dice = sortedDice;

    try {
        const response = await postJsonRequest(API_ENDPOINTS.SUGGEST_OPTIMAL_ACTION, currentData);
        if (!response) return;

        console.log("Optimal action response:", response);

        // Update gameState
        if (response.expected_value != null) {
            activePlayer.optimalActionResult = response.expected_value; // Store optimal expected value
        }

        if (currentData.rerolls_remaining > 0 && response.best_reroll) {
            const reversedMask = reverseMapping(response.best_reroll.mask_binary, mapping);
            activePlayer.optimalRerollMask = reversedMask.split("");
        } else if (currentData.rerolls_remaining === 0 && response.best_category) {
            activePlayer.optimalCategoryId = response.best_category.id;
        }
    } catch (error) {
        console.error("Error refreshing optimal action:", error);
    }
}

export function refreshUserAction() {
    console.log("Refreshing user action...");
    const activePlayer = gameState.getActivePlayer();
    const userAction = {};
    if (activePlayer.rerollsRemaining > 0) {
        const mask = activePlayer.diceState
            .map(die => (die.isLocked ? "0" : "1"))
            .join("");
        userAction.best_reroll = { mask_binary: mask };
    } else if (!userAction.best_category) {
        userAction.best_category = null;
    }
    // Update gameState with the new user evaluation
    activePlayer.userAction = userAction;
}

export async function refreshUserActionResult() {
    console.log("Refreshing user action result...");
    const activePlayer = gameState.getActivePlayer();

    if (!activePlayer) {
        console.warn("No active player found.");
        return;
    }

    const userAction = activePlayer.userAction?.best_reroll || activePlayer.userAction?.best_category;
    if (!userAction) {
        console.warn("No user action chosen.");
        activePlayer.userActionResult = { error: "Please evaluate an option." };
        return;
    }

    let diceToSend = activePlayer.diceState.map(die => die.value);
    if (diceToSend.length !== 5) {
        console.error("Invalid dice array length.");
        activePlayer.userActionResult = { error: "Invalid dice array length." };
        return;
    }

    let newUserAction;
    if (activePlayer.rerollsRemaining > 0) {
        const { sortedDice, mapping } = createMappingForSortedDice(diceToSend);
        const newMask = applyMappingToRerollMask(userAction.mask_binary, mapping);

        diceToSend = sortedDice;
        newUserAction = {
            best_reroll: {
                mask_binary: newMask,
                id: convertMaskToId(newMask),
            },
        };
    } else {
        newUserAction = {
            best_category: {
                id: userAction.id,
            },
        };
    }

    const payload = {
        upper_score: gameState.getUpperScore(),
        scored_categories: gameState.getScoredCategoriesBitmask(),
        dice: diceToSend,
        rerolls_remaining: activePlayer.rerollsRemaining,
        user_action: newUserAction,
    };

    try {
        const response = await postJsonRequest(API_ENDPOINTS.EVALUATE_USER_ACTION, payload);
        if (!response) return;

        console.log("Evaluation response:", response);

        activePlayer.userActionResult = {
            expectedValue: response.expected_value || null,
            details: response,
        };
    } catch (error) {
        console.error("Error evaluating user action:", error);
        activePlayer.userActionResult = { error: error.message };
    }
}
export async function refreshHistogram() {
    console.log("Refreshing histogram...");

    const activePlayer = gameState.getActivePlayer();
    if (!activePlayer || activePlayer.rerollsRemaining === 0) {
        console.log("Skipping histogram update as rerolls_remaining is 0.");
        return;
    }

    try {
        // Always recalculate total score
        const totalScore = gameState.getTotalScore();
        console.log("Current Total Score (recalculated):", totalScore);

        // Invalidate cache to ensure updated data is used
        const actionResponse = await postJsonRequest(API_ENDPOINTS.EVALUATE_ACTIONS, gameState.getEvaluationPayload());
        const histogramResponse = await fetch(API_ENDPOINTS.GET_HISTOGRAM).then(res => res.json());

        if (!actionResponse?.actions || !histogramResponse?.bins) {
            console.warn("Invalid response data for histogram.");
            return;
        }

        const { bins: optimalBins, min_ev, max_ev, bin_count } = histogramResponse;
        const binWidth = (max_ev - min_ev) / bin_count;

        const allEvs = [];
        const currentMaskEvs = [];
        const currentRerollMask = activePlayer.diceState
            .map(die => (die.isLocked ? "0" : "1"))
            .join("");

        actionResponse.actions.forEach(action => {
            if (action.distribution) {
                action.distribution.forEach(dist => {
                    const ev = dist.ev + totalScore;
                    allEvs.push(ev);

                    if (action.binary === currentRerollMask) {
                        currentMaskEvs.push(ev);
                    }
                });
            }
        });

        if (allEvs.length === 0) {
            console.warn("No expected values found in response.");
            return;
        }

        const currentBins = Array(bin_count).fill(0);
        const currentMaskBins = Array(bin_count).fill(0);

        allEvs.forEach(val => {
            const index = Math.min(Math.floor((val - min_ev) / binWidth), bin_count - 1);
            if (index >= 0) currentBins[index]++;
        });

        currentMaskEvs.forEach(val => {
            const index = Math.min(Math.floor((val - min_ev) / binWidth), bin_count - 1);
            if (index >= 0) currentMaskBins[index] += 2; // Scale for visibility
        });

        const totalOptimal = optimalBins.reduce((sum, val) => sum + val, 0);
        const reverseCumulativeBins = [];
        optimalBins.reduceRight((acc, val) => {
            const reverseCumulativeValue = acc + val;
            reverseCumulativeBins.unshift(reverseCumulativeValue / totalOptimal);
            return reverseCumulativeValue;
        }, 0);

        const labels = Array.from({ length: bin_count }, (_, i) => {
            const start = Math.round(min_ev + i * binWidth);
            const end = Math.round(start + binWidth);
            return `${start} - ${end}`;
        });
        activePlayer.histogramData = { currentBins, currentMaskBins, reverseCumulativeBins, labels };
    } catch (error) {
        console.error("Error refreshing histogram:", error);
    }
}