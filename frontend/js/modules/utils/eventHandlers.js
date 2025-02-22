import {postJsonRequest, API_ENDPOINTS} from "./endpoints.js";
import {runRefreshers} from "./refreshers.js";
import {renderUI} from "./uiBuilders.js";
import gameState from "../game/gameState.js";
import elements from "./elements.js";

export function attachEventHandlers() {
    document.body.addEventListener("click", async (event) => {
        const button = event.target.closest("button");
        if (!button) return;

        // For elements of which there are several we use classList to determine the action
        if (button.classList.contains("die-button") || button.classList.contains("dice-reroll-button")) {
            handleDiceActions(button).then(() =>
                console.log("Handled dice action.")
            );
            return;
        }

        if (button.classList.contains("evaluate-category")) {
            const categoryId = parseInt(button.dataset.catId, 10);
            if (!isNaN(categoryId)) handleEvaluateCategory(categoryId).then(() =>
                console.log(`Evaluated category ${categoryId}.`)
            );
            return;
        }

        // For elements with unique IDs we use the ID to determine the action
        switch (button.id) {
            case "addPlayerButton":
                handleAddPlayer();
                break;
            case "resetGameButton":
                handleResetGame();
                break;
            case "rerollIncrease":
            case "rerollDecrease":
                handleRerollActions(button);
                break;
            case "rerollDiceButton":
                await handleDiceReroll();
                break;
            case "randomizeDiceButton":
                await handleDiceRandomize();
                break;
            case "toggleOptimalActionButton":
                handleOptimalActionToggle();
                break;
            case "toggleDebugButton":
                handleDebugToggle();
                break;
            case "toggleInstructionsButton":
                handleInstructionToggle(button);
                break;
            default:
                console.warn(`Unhandled button click: ${button.id}`);
        }
    });

    elements.playerSelect.addEventListener("change", () => {
        const currentPlayer = elements.playerSelect.value;
        handlePlayerSelectionChange(currentPlayer);
    });

    elements.scorecardTable.addEventListener("change", (event) => {
        const checkbox = event.target.closest(".scored-check");
        if (checkbox) handleScoreCheckboxChange(event);
    });
}

function handlePlayerSelectionChange(selectedPlayerIndex) {
    gameState.setActivePlayerIndex(parseInt(selectedPlayerIndex, 10));
    gameState.saveState();
    runRefreshers().then(() => {
        renderUI();
    });
}

function handleAddPlayer() {
    gameState.addPlayer();
    const newPlayerIndex = gameState.players.length - 1;
    handlePlayerSelectionChange(newPlayerIndex);
    renderUI();
}

async function handleDiceRandomize() {
    await gameState.randomizeDice();
    gameState.saveState();
    runRefreshers().then(() => {
        renderUI();
    });
}

async function handleDiceActions(button) {
    const action = button.dataset.action;
    const index = parseInt(button.dataset.index, 10);

    if (isNaN(index) || !action) return;

    switch (action) {
        case "increment":
            await gameState.updateDieValue(index, Math.min(6, gameState.getActivePlayerDiceState()[index].value + 1));
            break;
        case "decrement":
            await gameState.updateDieValue(index, Math.max(1, gameState.getActivePlayerDiceState()[index].value - 1));
            break;
        case "toggle":
            await gameState.toggleDieLockState(index);
            break;
        default:
            console.warn(`Unhandled dice action: ${action}`);
    }
    runRefreshers().then(() => {
        gameState.saveState();
        renderUI();
    });
}

function handleRerollActions(button) {
    const activePlayer = gameState.getActivePlayer();
    if (!activePlayer) return;
    if (button.id === "rerollDecrease") {
        activePlayer.rerollsRemaining = Math.max(0, activePlayer.rerollsRemaining - 1);
    } else if (button.id === "rerollIncrease") {
        activePlayer.rerollsRemaining = Math.min(2, activePlayer.rerollsRemaining + 1);
    }
    runRefreshers().then(() => {
        gameState.saveState();
        renderUI();
    });
}

function handleInstructionToggle() {
    gameState.toggleInstructions();
    renderUI();
}

function handleOptimalActionToggle() {
    gameState.toggleOptimalAction();
    renderUI();
}

function handleDebugToggle() {
    const debugState = gameState.uiState.showDebugInfo;
    gameState.uiState.showDebugInfo = !debugState;
    gameState.saveState();
    renderUI();
}

let isResetting = false; // Flag to prevent multiple resets

function handleResetGame() {
    if (isResetting) {
        console.warn("Reset is already in progress.");
        return;
    }

    const confirmReset = window.confirm(
        "Are you sure you want to reset the game? This will erase all progress for all players."
    );
    if (!confirmReset) {
        console.log("Game reset canceled.");
        return;
    }

    // Set the flag to prevent further resets
    isResetting = true;

    try {
        console.log("Resetting game state...");
        gameState.resetGame();
        renderUI(); // Re-render UI after reset
    } catch (error) {
        console.error("Error during game reset:", error);
    } finally {
        // Reset the flag once the operation is complete
        isResetting = false;
    }
}

async function handleDiceReroll() {
    await gameState.rerollDice();
    await runRefreshers(); // Refresh available categories after reroll
    renderUI();
}

async function handleEvaluateCategory(categoryId) {
    const payload = gameState.getEvaluationPayload(categoryId);
    if (!payload) {
        console.error("Failed to generate evaluation payload.");
        return;
    }

    if (payload.rerolls_remaining !== 0) {
        console.warn("Cannot evaluate category while rerolls remain.");
        return;
    }

    try {
        const response = await postJsonRequest(API_ENDPOINTS.EVALUATE_USER_ACTION, payload);
        if (!response) return;

        console.log(`Evaluation result for category ${categoryId}:`, response);

        // Update the evaluation result in the active player's state
        const activePlayer = gameState.getActivePlayer();
        if (activePlayer) {
            activePlayer.userActionResult = {
                expectedValue: response.expected_value || null,
                details: response,
            };

            gameState.saveState();
            renderUI();
        } else {
            console.error("Active player not found.");
        }
    } catch (error) {
        console.error("Failed to evaluate user action:", error);
    }
}

function handleScoreCheckboxChange(event) {
    const checkbox = event.target;
    const catId = parseInt(checkbox.dataset.catId, 10);

    const activePlayerScorecard = gameState.getActivePlayerScorecard();
    const category = activePlayerScorecard.find(cat => cat.id === catId);
    if (!category) return;

    if (checkbox.checked) {
        category.isScored = true;
        category.score = category.suggestedScore || 0;
        gameState.randomizeDice().then(() => {
        })
    } else {
        category.isScored = false;
        category.score = 0;
    }

    gameState.saveState();
    runRefreshers().then(() => {
        renderUI();
    })
}