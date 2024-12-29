/****************************************************************
 * Loading Overlay Controls
 ****************************************************************/
function showLoading() {
    const overlay = document.getElementById("loadingOverlay");
    if (overlay) {
        overlay.classList.remove("hidden");
    }
}

function hideLoading() {
    const overlay = document.getElementById("loadingOverlay");
    if (overlay) {
        overlay.classList.add("hidden");
    }
}

/****************************************************************
 * Dice Mapping Helpers (NEW)
 ****************************************************************/
/**
 * Creates a mapping for sorted dice: returns { sortedDice, mapping }.
 * The mapping array elements look like { sortedIndex, originalIndex }.
 */
function createMappingForSortedDice(diceValues) {
    // Pair dice values with their original indices
    const pairedDice = diceValues.map((value, index) => ({value, index}));
    // Sort by dice value
    pairedDice.sort((a, b) => a.value - b.value);

    // Create the mapping from originalIndex -> sortedIndex
    const mapping = pairedDice.map((pair, sortedIndex) => ({
        sortedIndex,
        originalIndex: pair.index
    }));

    // Create the array of sorted dice values
    const sortedDice = pairedDice.map(pair => pair.value);

    return {sortedDice, mapping};
}

/**
 * Apply the mapping to a mask string (e.g., "01001"), reordering the bits.
 */
function applyMappingToRerollMask(mask, mapping) {
    const reorderedMask = new Array(mask.length);
    mapping.forEach(({sortedIndex, originalIndex}) => {
        reorderedMask[sortedIndex] = mask[originalIndex];
    });
    return reorderedMask.join("");
}

/**
 * Reverse a mapped mask back to the original dice order.
 */
function reverseMapping(mask, mapping) {
    const originalMask = new Array(mask.length);
    mapping.forEach(({sortedIndex, originalIndex}) => {
        originalMask[originalIndex] = mask[sortedIndex];
    });
    return originalMask.join("");
}

/****************************************************************
 Single-File Yatzy UI with Multiplayer Support Integrated
 ****************************************************************/
/* Configuration */
const YAHTZEE_CATEGORIES = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "One Pair", "Two Pairs", "Three of a Kind", "Four of a Kind",
    "Small Straight", "Large Straight", "Full House", "Chance", "Yatzy"
];
const TOTAL_DICE = 5;
const API_BASE_URL = "http://localhost:8080";
const URL_AVAILABLE_CATEGORIES = `${API_BASE_URL}/available_categories`;
const URL_SUGGEST_OPTIMAL_ACTION = `${API_BASE_URL}/suggest_optimal_action`;
const URL_EVALUATE_USER_ACTION = `${API_BASE_URL}/evaluate_user_action`;
const URL_EVALUATE_ACTIONS = `${API_BASE_URL}/evaluate_actions`;

/* --- Multiplayer State --- */
let players = [];
let activePlayerIndex = 0;

/* Instead of global single diceValues etc., each player will have their own state.
We'll provide helper functions to read/write from the active player's state. */
function getActivePlayer() {
    if (activePlayerIndex < 0 || activePlayerIndex >= players.length) {
        console.warn("Invalid activePlayerIndex:", activePlayerIndex);
        return null;
    }
    return players[activePlayerIndex];
}

/* DOM References */
const playerSelect = document.getElementById("playerSelect");
const addPlayerButton = document.getElementById("addPlayerButton");
const scorecardBodyElement = document.querySelector("#scorecard tbody");
const diceContainerElement = document.getElementById("diceContainer");
const rerollCountInput = document.getElementById("rerollsRemaining");
const differenceBoxContent = document.getElementById("differenceContent");
const optimalActionContainer = document.getElementById("optimalActionBox");
const optimalActionContentElement = document.getElementById("optimalActionContent");
const toggleOptimalButton = document.getElementById("toggleOptimalActionButton");
const debugInfoBox = document.getElementById("debugBox");
const toggleDebugButtonElement = document.getElementById("toggleDebugButton");
const rawOptimalActionTextArea = document.getElementById("rawOptimalAction");
const rawEvaluateActionTextArea = document.getElementById("rawEvaluateAction");
const userActionTextArea = document.getElementById("user_action");
const userActionEvaluationArea = document.getElementById("user_action_evaluation");

/* For backward compatibility, keep a global reference to diceValues.
We'll keep it in sync with the active player's dice state. */
let diceValues = [2, 2, 2, 5, 5];
let optimalRerollMask = [];
let currentOptimalExpectedValue = null;
let currentUserExpectedValue = null;
let currentOptimalCategoryID = null;

/****************************************************************
 * Dice Value Update - Single Pathway
 ****************************************************************/
function updateDiceValue(index, newValue) {
    const currentPlayer = getActivePlayer();
    if (!currentPlayer) {
        console.warn("No active player to update dice value.");
        return;
    }

    // Update the active player's dice array
    currentPlayer.diceValues[index] = newValue;

    // Keep the global diceValues in sync
    diceValues[index] = newValue;

    // Update the UI for the specific die
    const dieElement = diceContainerElement.querySelectorAll('.die')[index];
    if (dieElement) {
        const valueElement = dieElement.querySelector('.die-value');
        if (valueElement) {
            valueElement.textContent = newValue;
        }
    }
}

/* --- Player Management Logic --- */
function createDefaultPlayer(name) {
    return {
        name: name || `Player ${players.length + 1}`,
        diceValues: Array(TOTAL_DICE).fill(1),
        rerollsRemaining: 2,
        optimalRerollMask: Array(TOTAL_DICE).fill("0"),
        scorecard: []
    };
}

function addPlayer() {
    players.push(createDefaultPlayer(`Player ${players.length + 1}`));
    updatePlayerDropdown();
}

function updatePlayerDropdown() {
    playerSelect.innerHTML = players
        .map((p, index) => `<option value="${index}">${p.name}</option>`)
        .join("");
}

function savePlayerState() {
    const currentPlayer = getActivePlayer();
    if (!currentPlayer) return;

    // Save dice values (from UI)
    const dieValueDivs = diceContainerElement.querySelectorAll(".die-value");
    currentPlayer.diceValues = Array.from(dieValueDivs).map(div => parseInt(div.textContent, 10));

    // Save rerolls
    currentPlayer.rerollsRemaining = parseInt(rerollCountInput.value, 10);

    // Save scored categories
    currentPlayer.scorecard = Array.from(document.querySelectorAll(".scored-check")).map(chk => {
        const catId = parseInt(chk.dataset.catId, 10);
        const isChecked = chk.checked;
        const scoreCell = document.querySelector(`.best-score-cell[data-cat-id="${catId}"]`);
        const score = isChecked ? parseInt(scoreCell.textContent, 10) || 0 : 0;
        return {catId, isChecked, score};
    });
}

function loadPlayerState() {
    const currentPlayer = getActivePlayer();
    if (!currentPlayer) return;

    // If no scorecard, initialize
    if (!currentPlayer.scorecard) {
        currentPlayer.scorecard = [];
    }

    // Load dice values
    rerollCountInput.value = currentPlayer.rerollsRemaining;
    renderDiceUI();

    // Load scored categories
    document.querySelectorAll(".scored-check").forEach(chk => {
        const catId = parseInt(chk.dataset.catId, 10);
        const scoreCell = document.querySelector(`.best-score-cell[data-cat-id="${catId}"]`);
        const scoreData = currentPlayer.scorecard.find(item => item.catId === catId) || {};

        if (scoreData.isChecked) {
            chk.checked = true;
            scoreCell.textContent = scoreData.score || "0";
            const row = chk.closest("tr");
            row.classList.add("scored-category");
            row.classList.remove("available-category", "invalid-category");
            scoreCell.classList.add("locked");
        } else {
            chk.checked = false;
            scoreCell.textContent = "0";
            const row = chk.closest("tr");
            row.classList.remove("scored-category");
            scoreCell.classList.remove("locked");
        }
    });

    refreshUI();
}

/* Scorecard Creation */
function createScorecard() {
    YAHTZEE_CATEGORIES.forEach((category, i) => {
        const tr = document.createElement("tr");
        tr.dataset.catId = i;
        tr.innerHTML = `
                <td>${category}</td>
                <td>
                    <input type="checkbox" class="scored-check" data-cat-id="${i}">
                </td>
                <td class="best-score-cell" data-cat-id="${i}">0</td>
                <td>
                    <button class="evaluate-button" data-cat-id="${i}">Evaluate</button>
                </td>
            `;
        scorecardBodyElement.appendChild(tr);
    });

    // Bonus row (insert before index 6)
    const bonusTr = document.createElement("tr");
    bonusTr.classList.add("bonus-row");
    bonusTr.innerHTML = `
            <td>Upper Bonus</td>
            <td></td>
            <td class="best-score-cell bonus-best-score">0</td>
            <td></td>
        `;
    scorecardBodyElement.insertBefore(bonusTr, scorecardBodyElement.children[6]);

    // Total row (at the end)
    const totalTr = document.createElement("tr");
    totalTr.classList.add("total-row");
    totalTr.innerHTML = `
            <td>Total Score</td>
            <td></td>
            <td class="best-score-cell total-best-score">0</td>
            <td></td>
        `;
    scorecardBodyElement.appendChild(totalTr);
}

/* Dice UI Rendering */
function renderDiceUI() {
    const currentPlayer = getActivePlayer();
    if (!currentPlayer || !currentPlayer.diceValues) {
        console.warn("No active player or diceValues found.");
        return;
    }

    // Clear the dice container and rebuild the dice elements
    diceContainerElement.innerHTML = "";

    currentPlayer.diceValues.forEach((value, index) => {
        const dieDiv = document.createElement("div");
        dieDiv.classList.add("die", "locked"); // Default state
        dieDiv.innerHTML = `
            <button class="die-button" data-action="increment" data-index="${index}">▲</button>
            <div class="die-value">${value}</div>
            <button class="die-button" data-action="decrement" data-index="${index}">▼</button>
            <button data-action="toggle" data-index="${index}" class="dice-reroll-button locked">Locked</button>
        `;
        diceContainerElement.appendChild(dieDiv);
    });
}

function handleIncrement(index) {
    const currentPlayer = getActivePlayer();
    if (!currentPlayer) return;
    const currentValue = currentPlayer.diceValues[index];
    const newValue = Math.min(6, currentValue + 1);

    // Single pathway to update dice value
    updateDiceValue(index, newValue);

    refreshUserActionJSON();
    refreshAvailableCategories();
    refreshUI();
}

function handleDecrement(index) {
    const currentPlayer = getActivePlayer();
    if (!currentPlayer) return;
    const currentValue = currentPlayer.diceValues[index];
    const newValue = Math.max(1, currentValue - 1);

    // Single pathway to update dice value
    updateDiceValue(index, newValue);

    refreshUserActionJSON();
    refreshAvailableCategories();
    refreshUI();
}

/** Toggle locked/reroll state. */
function toggleDieState(index, buttonElement) {
    const currentPlayer = getActivePlayer();
    const dieContainer = buttonElement.closest(".die");
    if (!currentPlayer || !dieContainer) return;

    // Toggle the reroll/locked state
    if (buttonElement.classList.contains("locked")) {
        buttonElement.classList.remove("locked");
        buttonElement.classList.add("reroll");
        buttonElement.textContent = "Reroll";
        dieContainer.classList.remove("locked");
        dieContainer.classList.add("reroll");
        currentPlayer.optimalRerollMask[index] = "1";
    } else {
        buttonElement.classList.remove("reroll");
        buttonElement.classList.add("locked");
        buttonElement.textContent = "Locked";
        dieContainer.classList.remove("reroll");
        dieContainer.classList.add("locked");
        currentPlayer.optimalRerollMask[index] = "0";
    }

    // 1. Update the JSON representation of the user action
    refreshUserActionJSON();

    // 2. Fetch the optimal action based on the current state
    refreshOptimalAction();

    // 3. Evaluate the user action with the latest optimal action in mind
    evaluateUserActionSelection();

    // 4. Update the UI for valid scoring categories
    refreshAvailableCategories();

    // 5. Update the histogram to reflect the latest state
    updateHistogram();

    // 6. Perform a final UI refresh
    refreshUI();
}

/* Attach Event Handlers */
function attachEventHandlers() {
    // Centralized Event Listener for Dynamic and Static Elements
    document.body.addEventListener("click", (event) => {
        const button = event.target.closest("button");

        if (!button) return; // Ignore clicks outside buttons

        // Handle dice actions (increment, decrement, toggle)
        if (button.classList.contains("die-button") || button.classList.contains("dice-reroll-button")) {
            const action = button.dataset.action; // e.g., increment, decrement, toggle
            const index = parseInt(button.dataset.index, 10); // Dice index
            if (isNaN(index) || !action) return;

            switch (action) {
                case "increment":
                    handleIncrement(index);
                    break;
                case "decrement":
                    handleDecrement(index);
                    break;
                case "toggle":
                    toggleDieState(index, button);
                    break;
                default:
                    console.warn(`Unhandled dice action: ${action}`);
            }
            return;
        }

        // Handle Reroll actions
        if (button.id === "reroll_down" || button.id === "reroll_up") {
            const currentPlayer = getActivePlayer();
            if (!currentPlayer) return;

            if (button.id === "reroll_down") {
                currentPlayer.rerollsRemaining = Math.max(0, currentPlayer.rerollsRemaining - 1);
            } else if (button.id === "reroll_up") {
                currentPlayer.rerollsRemaining = Math.min(2, currentPlayer.rerollsRemaining + 1);
            }

            rerollCountInput.value = currentPlayer.rerollsRemaining;
            refreshAvailableCategories();
            refreshUI();
            return;
        }

        // Handle Evaluate Category actions
        if (button.classList.contains("evaluate-button")) {
            const categoryId = parseInt(button.dataset.catId, 10);
            if (isNaN(categoryId)) {
                console.warn("Invalid categoryId:", button.dataset.catId);
                return;
            }

            handleUserActionCategoryClick(categoryId);
            return;
        }

        // Handle other buttons by ID
        switch (button.id) {
            case "resetGameButton":
                const confirmReset = window.confirm("Are you sure you want to reset the game? This will erase all progress for all players.");
                if (confirmReset) {
                    resetGame();
                } else {
                    console.log("Game reset canceled.");
                }
                break;
            case "rerollDiceButton": {
                const currentPlayer = getActivePlayer();
                if (!currentPlayer) return;
                const r = currentPlayer.rerollsRemaining;
                if (r > 0) {
                    const diceDivs = document.querySelectorAll(".die");
                    diceDivs.forEach((div, idx) => {
                        const rerollBtn = div.querySelector(".dice-reroll-button");
                        if (rerollBtn.classList.contains("reroll")) {
                            const newVal = Math.floor(Math.random() * 6) + 1;
                            updateDiceValue(idx, newVal);
                        }
                    });
                    currentPlayer.rerollsRemaining = r - 1;
                    rerollCountInput.value = currentPlayer.rerollsRemaining;
                    refreshAvailableCategories();
                    refreshUI();
                }
            }
                break;
            case "randomizeDiceButton": {
                const player = getActivePlayer();
                if (!player) return;

                player.rerollsRemaining = 2;
                rerollCountInput.value = player.rerollsRemaining;

                const diceDivs = document.querySelectorAll(".die");
                diceDivs.forEach((div, index) => {
                    const newValue = Math.floor(Math.random() * 6) + 1;
                    updateDiceValue(index, newValue);

                    const rerollBtn = div.querySelector(".dice-reroll-button");
                    rerollBtn.classList.remove("reroll");
                    rerollBtn.classList.add("locked");
                    rerollBtn.textContent = "Locked";

                    div.classList.remove("reroll");
                    div.classList.add("locked");
                });

                refreshAvailableCategories();
                refreshUI();
            }
                break;
            case "toggleOptimalActionButton":
                handleOptimalActionToggle();
                break;
            case "toggleDebugButton":
                handleDebugToggle();
                break;
            default:
                console.warn(`Unhandled button click: ${button.id}`);
        }
    });

    // Player selection dropdown (static element)
    playerSelect.addEventListener("change", () => {
        savePlayerState();
        activePlayerIndex = parseInt(playerSelect.value, 10);
        loadPlayerState();
    });

    // Add Player button (static element)
    addPlayerButton.addEventListener("click", () => {
        savePlayerState();
        addPlayer();
        updatePlayerDropdown();
        activePlayerIndex = players.length - 1;
        playerSelect.value = activePlayerIndex;
        loadPlayerState();
    });

    // Scorecard checkbox events
    scorecardBodyElement.addEventListener("change", (event) => {
        const checkbox = event.target.closest(".scored-check");
        if (!checkbox) return;
        handleScoreCheckboxChange(event);
    });
}

/* Reset Game */
function resetGame() {
    localStorage.removeItem("yatzyGameState");

    // Reset global state
    players = [createDefaultPlayer("Player 1")];
    activePlayerIndex = 0;
    diceValues = Array(TOTAL_DICE).fill(1);
    optimalRerollMask = [];
    currentOptimalExpectedValue = null;
    currentUserExpectedValue = null;
    currentOptimalCategoryID = null;

    // Clear and rebuild UI
    scorecardBodyElement.innerHTML = "";
    createScorecard();
    updatePlayerDropdown();
    playerSelect.value = 0;
    rerollCountInput.value = 2;
    userActionEvaluationArea.innerHTML = "";
    differenceBoxContent.innerHTML = "Not available yet";
    rawOptimalActionTextArea.value = "";
    rawEvaluateActionTextArea.value = "";
    userActionTextArea.value = "";
    optimalActionContainer.classList.remove("show");
    toggleOptimalButton.textContent = "Show Optimal Action";
    optimalActionContainer.dataset.info = "None";
    optimalActionContentElement.innerHTML = "";
    debugInfoBox.classList.add("hidden");
    toggleDebugButtonElement.textContent = "Show Debug Info";
    const gameStatus = document.getElementById("game_status");
    if (gameStatus) {
        gameStatus.innerHTML = "";
    }

    // Rebuild dice UI
    renderDiceUI();

    // Randomize dice for fresh start
    document.getElementById("randomizeDiceButton").click();

    // Final UI refresh
    refreshUI();
}

/* Check if game is over */
function checkIfGameIsOver() {
    const scoredCount = document.querySelectorAll(".scored-check:checked").length;
    return scoredCount === YAHTZEE_CATEGORIES.length;
}

/* Score handling */
function handleScoreCheckboxChange(event) {
    const checkbox = event.target; // Get the event target (the checkbox element)
    const catId = parseInt(checkbox.dataset.catId, 10); // Extract the category ID
    const scoreCell = document.querySelector(`.best-score-cell[data-cat-id="${catId}"]`);
    const row = checkbox.closest("tr");

    if (checkbox.checked) {
        const bestScoreValue = parseInt(scoreCell.textContent, 10) || 0;
        scoreCell.textContent = bestScoreValue;
        scoreCell.classList.add("locked");
        row.classList.add("scored-category");
        row.classList.remove("available-category", "invalid-category");

        getActivePlayer().rerollsRemaining = 2;
        rerollCountInput.value = getActivePlayer().rerollsRemaining;

        userActionTextArea.value = JSON.stringify(
            {
                action: "scored",
                category: {id: catId, name: YAHTZEE_CATEGORIES[catId]},
                score: bestScoreValue,
            },
            null,
            2
        );
        console.log(userActionTextArea.value);
    } else {
        row.classList.remove("scored-category");
        row.classList.add("available-category");
        scoreCell.classList.remove("locked");

        userActionTextArea.value = JSON.stringify(
            {
                action: "unscored",
                category: {id: catId, name: YAHTZEE_CATEGORIES[catId]},
            },
            null,
            2
        );
    }

    recalculateScores();
    refreshAvailableCategories();
    refreshUI();
    saveGameState();
}

async function handleUserActionCategoryClick(categoryId) {
    const numericCategoryId = parseInt(categoryId, 10);
    if (isNaN(numericCategoryId)) {
        console.error("Invalid categoryId passed:", categoryId);
        return;
    }

    const currentPlayer = getActivePlayer();
    if (!currentPlayer) {
        console.error("No active player available.");
        return;
    }

    const rerollsRemaining = parseInt(rerollCountInput.value, 10);
    if (rerollsRemaining !== 0) {
        console.warn("Cannot evaluate category while rerolls remain.");
        return;
    }

    // Locate the row in the scorecard corresponding to the category
    const categoryRow = document.querySelector(`tr[data-cat-id="${numericCategoryId}"]`);
    if (!categoryRow || categoryRow.classList.contains("invalid-category")) {
        console.warn(`Category ${numericCategoryId} is invalid or not available.`);
        return;
    }

    // Prepare the payload for evaluation
    const payload = {
        upper_score: computeUpperScore(),
        scored_categories: collectScoredCategoriesBitmask(),
        dice: [...currentPlayer.diceValues],
        rerolls_remaining: rerollsRemaining,
        user_action: {
            best_category: {
                id: numericCategoryId,
                name: YAHTZEE_CATEGORIES[numericCategoryId],
            },
        },
    };

    try {
        // Call the evaluate_user_action endpoint
        const response = await sendJSONPostRequest(URL_EVALUATE_USER_ACTION, payload);
        if (!response) return;

        const data = response;
        console.log(`Evaluation result for category ${numericCategoryId}:`, data);

        // Update the UI with evaluation results
        userActionTextArea.value = JSON.stringify(data, null, 2);
        userActionEvaluationArea.innerHTML = `
                <p><b>Expected Total Score</b> for ${YAHTZEE_CATEGORIES[numericCategoryId]} = ${data.expected_value.toFixed(3)}</p>
            `;

        await refreshUI();

        // Optionally highlight the category row
        categoryRow.classList.add("evaluated-category");
    } catch (error) {
        console.error("Failed to evaluate user action:", error);
        userActionEvaluationArea.innerHTML = `
                <p>Error evaluating category: ${error.message}</p>
            `;
    }
}

/********************** NO DEBOUNCE BELOW **********************/

/* refreshAvailableCategories (direct call) */
async function refreshAvailableCategories() {
    const currentData = getCurrentGameData();
    if (!currentData) return;

    try {
        const raw = await sendJSONPostRequest(URL_AVAILABLE_CATEGORIES, currentData);
        if (!raw || !raw.categories) return;
        const data = raw;

        data.categories.forEach((c) => {
            const catId = c.id;
            const row = document.querySelector(`tr[data-cat-id="${catId}"]`);
            const chk = document.querySelector(`.scored-check[data-cat-id="${catId}"]`);
            const scoreCell = document.querySelector(`.best-score-cell[data-cat-id="${catId}"]`);

            if (chk.checked) {
                row.classList.add("scored-category");
                row.classList.remove("available-category", "invalid-category");
                scoreCell.classList.add("locked");
            } else if (c.valid) {
                row.classList.add("available-category");
                row.classList.remove("scored-category", "invalid-category");
                scoreCell.classList.remove("locked");
                scoreCell.textContent = c.score;
            } else {
                row.classList.add("invalid-category");
                row.classList.remove("scored-category", "available-category");
                scoreCell.classList.remove("locked");
                scoreCell.textContent = "0";
            }
        });
    } catch (e) {
        console.error("Error updating available categories:", e);
    }
}

/* refreshOptimalAction (direct call) */
async function refreshOptimalAction() {
    console.log("Refreshing optimal action...");
    document.querySelectorAll(".optimal-category").forEach((row) => row.classList.remove("optimal-category"));

    const currentPlayer = getActivePlayer();
    let s = getCurrentGameData();
    if (!s) return;

    // **IMPORTANT**: Sort dice before sending to the API
    const {sortedDice, mapping} = createMappingForSortedDice(s.dice);
    // Replace the original dice in 's' with sorted dice
    s.dice = sortedDice;

    try {
        const data = await sendJSONPostRequest(URL_SUGGEST_OPTIMAL_ACTION, s);
        if (!data) return;

        rawOptimalActionTextArea.value = JSON.stringify(data, null, 2);

        if (data.expected_value != null) {
            currentOptimalExpectedValue = parseFloat(data.expected_value);
            if (currentPlayer) {
                currentPlayer.lastOptimalExpectedValue = currentOptimalExpectedValue;
            }
        }

        let info = "None";
        if (s.rerolls_remaining > 0 && data.best_reroll) {
            // The mask returned is in sorted order; let's reverse it back to original
            const reversedMask = reverseMapping(data.best_reroll.mask_binary, mapping);

            info = `Reroll mask (sorted): ${data.best_reroll.mask_binary}<br>
                        Reroll mask (original): ${reversedMask}<br>
                        Expected Score: ${data.expected_value.toFixed(3)}`;
            optimalRerollMask = reversedMask.split("");

            if (currentPlayer) {
                currentPlayer.optimalRerollMask = [...optimalRerollMask];
            }
        } else if (s.rerolls_remaining === 0 && data.best_category) {
            info = `
                    <p>Optimal Category: <strong>${data.best_category.name}</strong></p>
                    <p>Expected Value: ${data.expected_value.toFixed(3)}</p>
                `;
            currentOptimalCategoryID = data.best_category.id;
            if (currentPlayer) {
                currentPlayer.currentOptimalCategoryID = currentOptimalCategoryID;
            }
        }

        optimalActionContainer.dataset.info = info;
        if (optimalActionContainer.classList.contains("show") && currentOptimalCategoryID != null) {
            const optimalRow = document.querySelector(`tr[data-cat-id='${currentOptimalCategoryID}']`);
            if (optimalRow) {
                optimalRow.classList.add("optimal-category");
            }
        }
        optimalActionContentElement.innerHTML = info;
    } catch (e) {
        rawOptimalActionTextArea.value = `Error: ${e.message}`;
    }
}

/* evaluateUserActionSelection (direct call) */
async function evaluateUserActionSelection() {
    const s = getCurrentGameData();
    if (!s) return;
    let ua = {};

    try {
        ua = JSON.parse(userActionTextArea.value);
    } catch (e) {
        userActionEvaluationArea.textContent = "No user action chosen or invalid JSON.";
        rawEvaluateActionTextArea.value = "No user action chosen or invalid JSON.";
        currentUserExpectedValue = null;
        if (getActivePlayer()) getActivePlayer().lastUserExpectedValue = null;
        return;
    }

    if (!ua || Object.keys(ua).length === 0) {
        userActionEvaluationArea.textContent = "No user action chosen.";
        rawEvaluateActionTextArea.value = "No user action chosen.";
        currentUserExpectedValue = null;
        if (getActivePlayer()) getActivePlayer().lastUserExpectedValue = null;
        return;
    }

    const currentPlayer = getActivePlayer();
    if (!currentPlayer) return;
    let diceToSend = [...currentPlayer.diceValues];
    let newUA = JSON.parse(JSON.stringify(ua));

    // If the user action includes a best_reroll AND rerolls > 0, we must send sorted dice
    // and reorder the mask accordingly
    if (s.rerolls_remaining > 0 && newUA.best_reroll && newUA.best_reroll.mask_binary) {
        const {sortedDice, mapping} = createMappingForSortedDice(diceToSend);
        const newMask = applyMappingToRerollMask(newUA.best_reroll.mask_binary, mapping);

        diceToSend = sortedDice;
        newUA.best_reroll.mask_binary = newMask;
        newUA.best_reroll.id = convertMaskToId(newMask);
    }

    // Replace the dice in s with the (possibly) sorted diceToSend
    const s2 = {...s, dice: diceToSend};

    try {
        const data = await sendJSONPostRequest(URL_EVALUATE_USER_ACTION, {
            ...s2,
            user_action: newUA,
        });
        if (!data) return;

        rawEvaluateActionTextArea.value = JSON.stringify(data, null, 2);

        if (data.expected_value != null) {
            currentUserExpectedValue = parseFloat(data.expected_value);
            const currentPlayer = getActivePlayer();
            if (currentPlayer) currentPlayer.lastUserExpectedValue = currentUserExpectedValue; // Optional

            let currentScore = 0;
            document.querySelectorAll(".scored-check:checked").forEach((checkbox) => {
                const catId = parseInt(checkbox.dataset.catId, 10);
                const scoreCell = document.querySelector(`.best-score-cell[data-cat-id="${catId}"]`);
                currentScore += parseInt(scoreCell.textContent, 10) || 0;
            });

            userActionEvaluationArea.innerHTML = `
                    <p><b>Expected Total Score = </b> ${(currentScore + currentUserExpectedValue).toFixed(3)}</p>
                `;
        } else {
            userActionEvaluationArea.textContent = "No user action chosen.";
            currentUserExpectedValue = null;
            if (currentPlayer) currentPlayer.lastUserExpectedValue = null;
        }
    } catch (e) {
        rawEvaluateActionTextArea.value = `Error: ${e.message}`;
        userActionEvaluationArea.textContent = `Error evaluating user action: ${e.message}`;
        currentUserExpectedValue = null;
        if (currentPlayer) currentPlayer.lastUserExpectedValue = null;
    } finally {
        await updateHistogram();
    }
}

async function refreshUI() {
    console.log("Refreshing UI...");
    if (checkIfGameIsOver()) {
        const totalScore = calculateFinalScore();
        document.getElementById("game_status").innerHTML = `
                <h2>Game Finished!</h2>
                <p>Total Score: <strong>${totalScore}</strong></p>
            `;
        localStorage.removeItem("yatzyGameState");
        return;
    }

    recalculateScores();
    refreshUserActionJSON();
    await refreshOptimalAction();
    await refreshAvailableCategories();
    await evaluateUserActionSelection();

    displaySortedDice();
    refreshRerollButton();
    refreshDifferenceBox();

    // Enable/Disable Evaluate buttons
    const r = getActivePlayer().rerollsRemaining;
    document.querySelectorAll(".evaluate-button").forEach((button) => {
        button.disabled = (r !== 0);
    });

    // Save state after every UI update
    saveGameState();
}

function calculateFinalScore() {
    let totalScore = 0;
    let upperScoreSum = 0;

    document.querySelectorAll(".scored-check:checked").forEach((checkbox) => {
        const catId = parseInt(checkbox.dataset.catId, 10);
        const scoreCell = document.querySelector(`.best-score-cell[data-cat-id="${catId}"]`);
        const scoreValue = parseInt(scoreCell.textContent, 10) || 0;
        totalScore += scoreValue;
        if (catId <= 5) {
            upperScoreSum += scoreValue;
        }
    });

    if (upperScoreSum >= 63) {
        totalScore += 50;
    }
    return totalScore;
}

function recalculateScores() {
    console.log("Recalculating scores...");
    let totalScore = 0;
    let upperScoreSum = 0;

    document.querySelectorAll(".scored-check").forEach((checkbox) => {
        if (checkbox.checked) {
            const catId = parseInt(checkbox.dataset.catId, 10);
            const scoreCell = document.querySelector(`.best-score-cell[data-cat-id="${catId}"]`);
            const scoreValue = parseInt(scoreCell.textContent, 10) || 0;

            totalScore += scoreValue;
            if (catId <= 5) {
                upperScoreSum += scoreValue;
            }
        }
    });

    const bonus = upperScoreSum >= 63 ? 50 : 0;
    const bonusCell = document.querySelector(".bonus-best-score");
    if (bonusCell) bonusCell.textContent = upperScoreSum;
    const totalCell = document.querySelector(".total-best-score");
    if (totalCell) totalCell.textContent = totalScore + bonus;
}

function refreshUserActionJSON() {
    console.log("Refreshing user action JSON...");
    const r = getActivePlayer().rerollsRemaining;
    let ua = {};
    try {
        ua = JSON.parse(userActionTextArea.value) || {};
    } catch (e) {
        ua = {};
    }

    if (r > 0) {
        let mask = "";
        const diceDivs = document.querySelectorAll(".die");
        diceDivs.forEach((div) => {
            const btn = div.querySelector(".dice-reroll-button");
            mask += btn.classList.contains("reroll") ? "1" : "0";
        });
        ua = {best_reroll: {mask_binary: mask}};
    } else if (!ua.best_category) {
        ua = {};
    }

    userActionTextArea.value = JSON.stringify(ua, null, 2);
}

function refreshRerollButton() {
    const r = getActivePlayer().rerollsRemaining;
    document.getElementById("rerollDiceButton").disabled = (r === 0);
}

function refreshDifferenceBox() {
    if (currentOptimalExpectedValue == null || currentUserExpectedValue == null) {
        differenceBoxContent.innerHTML = "Not available yet";
    } else {
        const diff = (currentOptimalExpectedValue - currentUserExpectedValue).toFixed(3);
        differenceBoxContent.innerHTML = "&Delta; = " + diff;
    }
}

/* State Saving / Restoration */
function saveGameState() {
    const gameState = getGameStateAsJSON();
    localStorage.setItem("yatzyGameState", JSON.stringify(gameState));
}

function getGameStateAsJSON() {
    savePlayerState(); // Make sure we have latest from UI

    const allPlayersState = players.map(p => ({
        name: p.name,
        diceValues: [...p.diceValues],
        optimalRerollMask: [...p.optimalRerollMask],
        lastOptimalExpectedValue: p.lastOptimalExpectedValue,
        lastUserExpectedValue: p.lastUserExpectedValue,
        currentOptimalCategoryID: p.currentOptimalCategoryID,
        rerollsRemaining: p.rerollsRemaining,
        scorecard: p.scorecard
    }));

    const gameState = {
        dice: [...diceValues],
        rerolls_remaining: parseInt(rerollCountInput.value, 10),
        scored_categories: collectScoredCategoriesBitmask(),
        scored_category_scores: collectScoredCategoryScores(),
        upper_score: computeUpperScore(),
        total_score: computeTotalScore(),
        optimal_best_reroll_mask: optimalRerollMask,
        lastOptimalEV: currentOptimalExpectedValue,
        lastUserEV: currentUserExpectedValue,
        currentOptimalCategoryId: currentOptimalCategoryID,
        players: allPlayersState,
        activePlayerIndex
    };
    return gameState;
}

function collectScoredCategoriesBitmask() {
    let bitmask = 0;
    document.querySelectorAll(".scored-check").forEach((checkbox) => {
        if (checkbox.checked) {
            const catId = parseInt(checkbox.dataset.catId, 10);
            bitmask |= 1 << catId;
        }
    });
    return bitmask;
}

function collectScoredCategoryScores() {
    const scores = {};
    document.querySelectorAll(".scored-check").forEach((checkbox) => {
        if (checkbox.checked) {
            const catId = parseInt(checkbox.dataset.catId, 10);
            const scoreCell = document.querySelector(`.best-score-cell[data-cat-id='${catId}']`);
            scores[catId] = scoreCell ? parseInt(scoreCell.textContent, 10) || 0 : 0;
        }
    });
    return scores;
}

function computeUpperScore() {
    let upperSum = 0;
    document.querySelectorAll(".scored-check:checked").forEach((checkbox) => {
        const catId = parseInt(checkbox.dataset.catId, 10);
        const scoreCell = document.querySelector(`.best-score-cell[data-cat-id='${catId}']`);
        const val = scoreCell ? parseInt(scoreCell.textContent, 10) || 0 : 0;
        if (catId <= 5) upperSum += val;
    });
    if (upperSum >= 63) return upperSum + 50;
    return upperSum;
}

function computeTotalScore() {
    let totalScore = 0;
    let upperSum = 0;
    document.querySelectorAll(".scored-check:checked").forEach((checkbox) => {
        const catId = parseInt(checkbox.dataset.catId, 10);
        const scoreCell = document.querySelector(`.best-score-cell[data-cat-id='${catId}']`);
        const val = scoreCell ? parseInt(scoreCell.textContent, 10) || 0 : 0;
        totalScore += val;
    });

    document.querySelectorAll(".scored-check:checked").forEach((checkbox) => {
        const catId = parseInt(checkbox.dataset.catId, 10);
        const scoreCell = document.querySelector(`.best-score-cell[data-cat-id='${catId}']`);
        const val = scoreCell ? parseInt(scoreCell.textContent, 10) || 0 : 0;
        if (catId <= 5) upperSum += val;
    });

    if (upperSum >= 63) totalScore += 50;
    return totalScore;
}

function restoreGameFromState(input) {
    try {
        const state = typeof input === "string" ? JSON.parse(input) : input;

        // Scorecard
        scorecardBodyElement.innerHTML = "";
        createScorecard();

        // Restore players
        if (state.players && Array.isArray(state.players)) {
            players = state.players.map(p => ({
                name: p.name,
                diceValues: Array.isArray(p.diceValues) ? [...p.diceValues] : Array(TOTAL_DICE).fill(1),
                optimalRerollMask: Array.isArray(p.optimalRerollMask) ? [...p.optimalRerollMask] : [],
                lastOptimalExpectedValue: p.lastOptimalExpectedValue || null,
                lastUserExpectedValue: p.lastUserExpectedValue || null,
                currentOptimalCategoryID: p.currentOptimalCategoryID || null,
                rerollsRemaining: typeof p.rerollsRemaining === 'number' ? p.rerollsRemaining : 2,
                scorecard: p.scorecard || []
            }));
            activePlayerIndex = state.activePlayerIndex || 0;
        }

        // UI and load
        updatePlayerDropdown();
        playerSelect.value = activePlayerIndex;
        loadPlayerState();

        recalculateScores();
        refreshAvailableCategories();
        refreshUI();

        return true;
    } catch (error) {
        console.error('Failed to restore game state:', error);
        return false;
    }
}

/* Category / Reroll / Optimal */
async function sendJSONPostRequest(url, data) {
    showLoading();
    console.log("Sending POST request to:", url, data);
    try {
        const response = await fetch(url, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error("API call failed:", error);
        alert("An error occurred. Please try again later.\n" + error.message);
        return null;
    } finally {
        hideLoading();
    }
}

let histogramChart = null;

async function getCachedHistogramData() {
    const CACHE_KEY = 'histogram_cache';
    const CACHE_DURATION = 60 * 60 * 1000;

    try {
        const cached = localStorage.getItem(CACHE_KEY);
        if (cached) {
            const {timestamp, data} = JSON.parse(cached);
            if (Date.now() - timestamp < CACHE_DURATION) {
                return data;
            }
        }
        const response = await fetch('http://localhost:8080/score_histogram');
        if (!response.ok) {
            throw new Error(`Histogram endpoint responded with ${response.status}`);
        }
        const data = await response.json();
        localStorage.setItem(CACHE_KEY, JSON.stringify({
            timestamp: Date.now(),
            data
        }));
        return data;
    } catch (error) {
        console.error('Error fetching histogram data:', error);
        return null;
    }
}

async function updateHistogram() {
    const currentData = getCurrentGameData();
    if (!currentData) return;

    // Skip updating if rerolls_remaining is 0
    if (currentData.rerolls_remaining === 0) {
        console.log("Skipping histogram update as rerolls_remaining is 0.");
        return; // Exit early, keeping the existing histogram
    }

    try {
        const [actionResponse, histogramResponse] = await Promise.all([
            sendJSONPostRequest(URL_EVALUATE_ACTIONS, currentData),
            getCachedHistogramData()
        ]);

        if (!actionResponse || !actionResponse.actions || !histogramResponse) {
            console.warn("No histogram data or invalid response from evaluate_actions");
            return;
        }

        const responseData = actionResponse;

        const allEvs = [];
        const currentMaskEvs = [];
        // Compute the current reroll mask based on the UI state of the dice
        const currentRerollMask = Array.from(document.querySelectorAll(".dice-reroll-button"))
            .map(button => button.classList.contains("reroll") ? "1" : "0")
            .join("");

        responseData.actions.forEach((action) => {
            if (action.distribution) {
                action.distribution.forEach((dist) => {
                    const ev = dist.ev + currentData.total_score_no_bonus;
                    allEvs.push(ev);

                    // Match EVs for the current reroll mask
                    if (action.binary === currentRerollMask) {
                        currentMaskEvs.push(ev);
                    }
                });
            }
        });

        if (allEvs.length === 0) {
            console.warn("No expected values found in response.");
            if (histogramChart) {
                histogramChart.destroy();
                histogramChart = null;
            }
            return;
        }

        const minEV = 100;
        const maxEV = 380;
        const binCount = 56;
        const binWidth = (maxEV - minEV) / binCount;
        const currentBins = Array(binCount).fill(0);
        const currentMaskBins = Array(binCount).fill(0);

        // Populate bins
        allEvs.forEach((val) => {
            const index = Math.min(Math.floor((val - minEV) / binWidth), binCount - 1);
            if (index >= 0) currentBins[index]++;
        });

        currentMaskEvs.forEach((val) => {
            const index = Math.min(Math.floor((val - minEV) / binWidth), binCount - 1);
            if (index >= 0) {
                // Arbitrarily scale these bins so they show up distinctly
                currentMaskBins[index] += 2;
            }
        });

        // Calculate reverse cumulative probability for the "10k Optimal" dataset
        const totalOptimal = histogramResponse.bins.reduce((sum, val) => sum + val, 0);
        const reverseCumulativeBins = [];
        histogramResponse.bins.reduceRight((acc, val) => {
            const reverseCumulativeValue = acc + val;
            reverseCumulativeBins.unshift(reverseCumulativeValue / totalOptimal);
            return reverseCumulativeValue;
        }, 0);

        const labels = Array.from({length: binCount}, (_, i) => {
            const start = Math.round(minEV + i * binWidth);
            const end = Math.round(start + binWidth);
            return `${start.toFixed(0)} - ${end.toFixed(0)}`;
        });

        const ctx = document.getElementById("histogramChart");
        if (!ctx) {
            console.error("Histogram canvas not found in DOM.");
            return;
        }

        const chartConfig = {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        type: "line",
                        label: "10k Optimal",
                        data: reverseCumulativeBins,
                        backgroundColor: "rgba(40, 150, 250, 0.1)",
                        borderColor: "rgba(40, 150, 250, 1)",
                        borderWidth: 1,
                        tension: 0.4,
                        yAxisID: "y1",
                        animation: false,
                    },
                    {
                        label: "Expected Scores Possible",
                        data: currentBins,
                        backgroundColor: "rgba(255, 100, 40, 0.9)",
                        borderColor: "rgba(255, 100, 40, 1)",
                        borderWidth: 1,
                        yAxisID: "y",
                        animation: false,
                    },
                    {
                        label: "Current Reroll Mask",
                        data: currentMaskBins,
                        backgroundColor: "rgba(50, 205, 50, 0.9)",
                        borderColor: "rgba(50, 205, 50, 1)",
                        borderWidth: 1,
                        yAxisID: "y",
                        animation: true,
                    },
                ],
            },
            options: {
                responsive: true,
                animation: {
                    duration: 200,
                },
                scales: {
                    x: {
                        title: {display: true, text: "Expected Scores"},
                        grid: {display: true},
                    },
                    y: {
                        type: "logarithmic",
                        title: {display: true, text: "Frequency (Log Scale)"},
                        min: 1,
                        max: 1000,
                        position: "left",
                        grid: {display: false},
                    },
                    y1: {
                        title: {display: true, text: "Reverse Cumulative Probability"},
                        min: 0,
                        max: 1,
                        position: "right",
                        grid: {drawOnChartArea: true},
                    },
                },
            },
        };

        if (!histogramChart) {
            histogramChart = new Chart(ctx.getContext("2d"), chartConfig);
        } else {
            histogramChart.data = chartConfig.data;
            histogramChart.update();
        }
    } catch (error) {
        console.error("Histogram update failed:", error);
    }
}

function getCurrentGameData() {
    const currentPlayer = getActivePlayer();
    if (!currentPlayer) {
        console.warn("No active player. Cannot retrieve game data.");
        return null;
    }

    // Calculate scored categories as a bitmask
    let bitmask = 0;
    let total = 0;
    let upperSum = 0;

    document.querySelectorAll(".scored-check").forEach((checkbox) => {
        if (checkbox.checked) {
            const catId = parseInt(checkbox.dataset.catId, 10);
            const scoreCell = document.querySelector(`.best-score-cell[data-cat-id='${catId}']`);
            const value = scoreCell ? parseInt(scoreCell.textContent, 10) || 0 : 0;
            bitmask |= 1 << catId;
            total += value;
            if (catId <= 5) {
                upperSum += value;
            }
        }
    });

    const bonus = upperSum >= 63 ? 50 : 0;

    return {
        upper_score: upperSum,
        scored_categories: bitmask,
        dice: [...currentPlayer.diceValues],
        rerolls_remaining: currentPlayer.rerollsRemaining,
        bonus: bonus,
        total_score_no_bonus: total,
        total_score: total + bonus,
    };
}

function convertMaskToId(mask) {
    let id = 0;
    for (let i = 0; i < mask.length; i++) {
        if (mask[i] === "1") {
            id |= 1 << i;
        }
    }
    return id;
}

function displaySortedDice() {
    const currentPlayer = getActivePlayer();
    const container = document.getElementById("sorted_dice_container");
    if (!currentPlayer || !container) {
        console.warn("No active player or container found.");
        return;
    }

    container.innerHTML = "";

    // Get dice values and the reroll mask
    const diceValues = [...currentPlayer.diceValues];
    const rerollMask = currentPlayer.optimalRerollMask || [];

    // Create an array of dice with their original indices
    const sortedDice = diceValues.map((value, index) => ({value, index})).sort((a, b) => a.value - b.value);

    // Add sorted dice to the container
    sortedDice.forEach(({value, index}) => {
        const diceDiv = document.createElement("div");
        diceDiv.classList.add("die-value");
        diceDiv.textContent = value;

        // Highlight dice to be rerolled
        if (rerollMask[index] === "1") {
            diceDiv.classList.add("reroll-highlight");
        }

        container.appendChild(diceDiv);
    });
}

function handleOptimalActionToggle() {
    if (optimalActionContainer.classList.contains("show")) {
        // Hide Optimal Action box
        optimalActionContainer.classList.remove("show");
        toggleOptimalButton.textContent = "Show Optimal Action";

        // Clear any highlighted rows for optimal categories
        document.querySelectorAll(".optimal-category").forEach(row => {
            row.classList.remove("optimal-category");
        });
    } else {
        // Show Optimal Action box
        optimalActionContainer.classList.add("show");
        toggleOptimalButton.textContent = "Hide Optimal Action";

        // Display optimal action information
        let info = optimalActionContainer.dataset.info || "None";
        optimalActionContentElement.innerHTML = info;

        // Highlight the optimal category row if one exists
        if (currentOptimalCategoryID != null) {
            const optimalRow = document.querySelector(`tr[data-cat-id='${currentOptimalCategoryID}']`);
            if (optimalRow) {
                optimalRow.classList.add("optimal-category");
            }
        }

        // Display sorted dice and highlight those to be rerolled
        displaySortedDice();
    }
}

function handleDebugToggle() {
    if (debugInfoBox.classList.contains("hidden")) {
        // Show Debug Info
        debugInfoBox.classList.remove("hidden");
        toggleDebugButtonElement.textContent = "Hide Debug Info";
    } else {
        // Hide Debug Info
        debugInfoBox.classList.add("hidden");
        toggleDebugButtonElement.textContent = "Show Debug Info";
    }
}

function init() {
    // Ensure at least one default player
    if (players.length === 0) {
        players.push(createDefaultPlayer("Player 1"));
    }
    activePlayerIndex = 0;

    // Build UI
    createScorecard();
    renderDiceUI();

    attachEventHandlers();

    // Populate player dropdown
    updatePlayerDropdown();
    playerSelect.value = activePlayerIndex;

    // Attempt restore
    const savedState = localStorage.getItem("yatzyGameState");
    if (savedState) {
        try {
            restoreGameFromState(savedState);
            return;
        } catch (error) {
            console.error("Failed to load saved state:", error);
            localStorage.removeItem("yatzyGameState");
        }
    }

    // Fresh random dice if no restore
    document.getElementById("randomizeDiceButton").click();
    refreshUI();
}

window.addEventListener('beforeunload', () => {
    if (!checkIfGameIsOver()) {
        saveGameState();
    }
});

document.addEventListener("DOMContentLoaded", () => {
    init();
    document.body.classList.add("loaded");
});