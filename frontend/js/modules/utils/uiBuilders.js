import gameState from "../game/gameState.js";
import elements from "./elements.js";
import {getSortedDiceWithHighlights} from "./mappings.js";
import {createChartConfig} from "./chartConfig.js";

export function renderUI() {
    console.log("Rendering UI...");
    renderUserEvaluation();
    renderScorecard();
    renderPlayerDropdown();
    renderDice();
    renderRerollsRemaining();
    renderRerollButton();
    renderDelta();
    renderHistogram();
    renderSortedDice();
    renderEvaluateButtons();
    renderOptimalAction();
    renderDebugInfo();
    renderInstructions();
}

export function renderEvaluateButtons() {
    const activePlayer = gameState.getActivePlayer();
    elements.evaluateButtons.forEach((button) => {
        button.disabled = activePlayer.rerollsRemaining !== 0;
    });
}

export function renderRerollButton() {
    const activePlayer = gameState.getActivePlayer();
    elements.rerollButton.disabled = activePlayer.rerollsRemaining === 0;
}

export function renderRerollsRemaining() {
    const activePlayer = gameState.getActivePlayer();
    if (!activePlayer) {
        console.warn("No active player found.");
        return;
    }
    elements.rerollsRemaining.value = activePlayer.rerollsRemaining;
}

export function renderDelta() {
    const activePlayer = gameState.getActivePlayer();

    if (!elements.deltaBox) {
        console.warn("Delta box element not found.");
        return;
    }

    if (!activePlayer) {
        elements.deltaBox.innerHTML = "Not available yet";
        return;
    }

    const optimalActionResult = activePlayer.optimalActionResult;
    const userActionResult = activePlayer.userActionResult?.expectedValue || null;

    if (optimalActionResult == null || userActionResult == null) {
        elements.deltaBox.innerHTML = "Not available yet";
        return;
    }

    const delta = (optimalActionResult - userActionResult).toFixed(3);
    elements.deltaBox.innerHTML = `&Delta; = ${delta}`;
}

export function renderSortedDice() {
    const sortedDice = getSortedDiceWithHighlights();

    if (!elements.sortedDiceContainer || !sortedDice) {
        console.warn("No active player or sorted dice data found.");
        return;
    }

    elements.sortedDiceContainer.innerHTML = sortedDice
        .map(die => `
            <div class="die-value ${die.isHighlighted ? "reroll-highlight" : ""}">
                ${die.value}
            </div>
        `).join("");
}

function renderUserEvaluation() {
    const activePlayer = gameState.getActivePlayer();

    if (!elements.userEvaluation) {
        console.warn("User evaluation element not found.");
        return;
    }

    // Clear the previous content explicitly
    elements.userEvaluation.innerHTML = "";

    if (!activePlayer || !activePlayer.userActionResult) {
        elements.userEvaluation.innerHTML = `<p>No evaluation available.</p>`;
        return;
    }

    const { expectedValue, details, error } = activePlayer.userActionResult;

    if (error) {
        elements.userEvaluation.innerHTML = `<p>Error: ${error}</p>`;
    } else {
        // Add current total score to expected value
        const totalScore = gameState.getTotalScore();
        const bonus = gameState.getBonus();
        const adjustedExpectedValue = expectedValue != null ? (expectedValue + totalScore + bonus).toFixed(3) : "N/A";

        elements.userEvaluation.innerHTML = `
            <p><b>Expected Total Score:</b> ${adjustedExpectedValue}</p>
            <pre>${JSON.stringify(details, null, 2)}</pre>
        `;
    }
}

export function renderScorecard() {
    const activePlayer = gameState.getActivePlayer();
    const activePlayerScorecard = gameState.getActivePlayerScorecard();

    if (!elements.scorecardTable) {
        console.warn("Scorecard table element not found.");
        return;
    }

    const tbody = elements.scorecardTable.querySelector("tbody");
    if (!tbody) {
        console.warn("Scorecard tbody not found.");
        return;
    }

    // Clear the existing content in the tbody
    tbody.innerHTML = "";

    // Separate upper and lower categories
    const upperCategories = activePlayerScorecard.filter(cat => cat.id <= 5); // Ones to Sixes
    const lowerCategories = activePlayerScorecard.filter(cat => cat.id > 5); // Everything else

    // Render upper categories
    upperCategories.forEach(category => {
        const tr = document.createElement("tr");
        tr.dataset.catId = category.id;

        // Add classes for styling based on category validity and scoring status
        tr.className = "";
        if (category.isValid) tr.classList.add("available-category");
        if (!category.isValid) tr.classList.add("invalid-category");
        if (category.isScored) tr.classList.add("scored-category");

        tr.innerHTML = `
            <td>${category.name}</td>
            <td>
                <input 
                    type="checkbox" 
                    class="scored-check" 
                    data-cat-id="${category.id}" 
                    ${category.isScored ? "checked" : ""}>
            </td>
            <td class="score" data-cat-id="${category.id}">
                ${category.isScored
            ? category.score
            : category.suggestedScore || 0}
            </td>
            <td>
                <button 
                    class="evaluate-category" 
                    data-cat-id="${category.id}" 
                    ${category.isValid && activePlayer.rerollsRemaining === 0 ? "" : "disabled"}>
                    Evaluate
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });

// Add bonus row after upper categories
    const upperScore = gameState.getUpperScore(); // Get the upper score
    const bonus = gameState.getBonus(); // Get the bonus (50 or 0)

    // Add upper score row
    const upperScoreRow = `
    <tr class="bonus-row">
        <td>Upper Score</td>
        <td></td>
        <td class="score bonus-score">${upperScore}</td>
        <td></td>
    </tr>`;
    tbody.insertAdjacentHTML("beforeend", upperScoreRow);

// Add bonus row
    const bonusRow = `
    <tr class="bonus-row">
        <td>Bonus</td>
        <td class="bonus-value">${bonus > 0 ? "🤩" : ""}</td>
        <td class="bonus-value">${bonus > 0 ? "50" : ""}</td>
        <td></td>
    </tr>`;
    tbody.insertAdjacentHTML("beforeend", bonusRow);

    // Render lower categories
    lowerCategories.forEach(category => {
        const tr = document.createElement("tr");
        tr.dataset.catId = category.id;

        tr.className = "";
        if (category.isValid) tr.classList.add("available-category");
        if (!category.isValid) tr.classList.add("invalid-category", "gray-category");
        if (category.isScored) tr.classList.add("scored-category");

        tr.innerHTML = `
            <td>${category.name}</td>
            <td>
                <input 
                    type="checkbox" 
                    class="scored-check" 
                    data-cat-id="${category.id}" 
                    ${category.isScored ? "checked" : ""}>
            </td>
            <td class="score" data-cat-id="${category.id}">
                ${category.isScored
            ? category.score
            : category.suggestedScore || 0}
            </td>
            <td>
                <button 
                    class="evaluate-category" 
                    data-cat-id="${category.id}" 
                    ${category.isValid && activePlayer.rerollsRemaining === 0 ? "" : "disabled"}>
                    Evaluate
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });

    // Add total row
    const totalRow = `
        <tr class="total-row">
            <td>Total Score</td>
            <td></td>
            <td class="score total-score">${gameState.getTotalScore() + gameState.getBonus()}</td>
            <td></td>
        </tr>`;
    tbody.insertAdjacentHTML("beforeend", totalRow);
}

export function renderDice() {
    const activePlayer = gameState.getActivePlayer();

    if (!activePlayer || !activePlayer.diceState) {
        console.warn("No active player or diceState found.");
        return;
    }

    // Clear the dice container and rebuild the dice elements
    if (!elements.diceContainer) {
        console.warn("Dice container element not found.");
        return;
    }

    elements.diceContainer.innerHTML = "";
    activePlayer.diceState.forEach((die, index) => {
        const dieDiv = document.createElement("div");
        dieDiv.classList.add("die", die.isLocked ? "locked" : "reroll");

        dieDiv.innerHTML = `
            <button class="die-button" data-action="increment" data-index="${index}">▲</button>
            <div class="die-value">${die.value}</div>
            <button class="die-button" data-action="decrement" data-index="${index}">▼</button>
            <button 
                class="dice-reroll-button ${die.isLocked ? "locked" : "reroll"}" 
                data-action="toggle" 
                data-index="${index}">
                ${die.isLocked ? "Locked" : "Reroll"}
            </button>
        `;

        elements.diceContainer.appendChild(dieDiv);
    });
}

export function renderPlayerDropdown() {
    const activePlayerIndex = gameState.activePlayerIndex;

    elements.playerSelect.innerHTML = gameState.players
        .map((player, index) => `<option value="${index}">${player.name}</option>`)
        .join("");

    elements.playerSelect.value = activePlayerIndex; // Set the dropdown to the active player
}

function renderOptimalAction() {
    const { showOptimalAction } = gameState.uiState;
    if (showOptimalAction) {
        elements.optimalActionBox.classList.add("show");
        elements.optimalActionButton.textContent = "Hide Optimal Action";
    } else {
        elements.optimalActionBox.classList.remove("show");
        elements.optimalActionButton.textContent = "Show Optimal Action";
    }
}

function autoResizeTextarea(textarea) {
    textarea.style.height = "auto"; // Reset height
    textarea.style.height = `${textarea.scrollHeight}px`; // Set height to fit content
}


function renderDebugInfo() {
    const { showDebugInfo } = gameState.uiState;
    const debugBox = document.getElementById("debugBox");
    const debugButton = document.getElementById("toggleDebugButton");
    const rawOptimalAction = document.getElementById("rawOptimalAction");
    const rawEvaluateAction = document.getElementById("rawEvaluateAction");

    if (!debugBox || !debugButton || !rawOptimalAction || !rawEvaluateAction) {
        console.warn("Debug box or textareas not found.");
        return;
    }

    // Update the visibility of the debug box
    if (showDebugInfo) {
        debugBox.classList.remove("hidden");
        debugButton.textContent = "Hide Debug Info";
    } else {
        debugBox.classList.add("hidden");
        debugButton.textContent = "Show Debug Info";
    }

    // Populate the debug info with JSON data
    const activePlayer = gameState.getActivePlayer();
    if (activePlayer) {
        const optimalAction = activePlayer.optimalActionResult || {};
        const userAction = activePlayer.userAction || {};

        rawOptimalAction.value = JSON.stringify(optimalAction, null, 2);
        rawEvaluateAction.value = JSON.stringify(userAction, null, 2);
    } else {
        rawOptimalAction.value = "No data available.";
        rawEvaluateAction.value = "No data available.";
    }

    // Dynamically resize textareas to fit content
    autoResizeTextarea(rawOptimalAction);
    autoResizeTextarea(rawEvaluateAction);
}

function renderInstructions() {
    const { showInstructions } = gameState.uiState;
    if (showInstructions) {
        elements.instructionsBox.classList.add("show");
        elements.instructionsButton.textContent = "Hide Instructions";
    } else {
        elements.instructionsBox.classList.remove("show");
        elements.instructionsButton.textContent = "Show Instructions";
    }
}

export function renderHistogram() {
    const activePlayer = gameState.getActivePlayer();
    if (!activePlayer || !activePlayer.histogramData || !elements.histogramCanvas) {
        console.warn("No histogram data or canvas element found.");
        return;
    }

    const { currentBins = [], currentMaskBins = [], reverseCumulativeBins = [], labels = [] } = activePlayer.histogramData;

    if (currentBins.length === 0 || reverseCumulativeBins.length === 0) {
        console.warn("Histogram data is incomplete.");
        return;
    }

    const chartConfig = createChartConfig(labels, reverseCumulativeBins, currentBins, currentMaskBins);

    if (window.histogramChart instanceof Chart) {
        window.histogramChart.data = chartConfig.data;
        window.histogramChart.update();
    } else {
        if (window.histogramChart) {
            console.warn("Invalid histogramChart instance. Destroying...");
            window.histogramChart.destroy();
        }

        window.histogramChart = new Chart(elements.histogramCanvas.getContext("2d"), chartConfig);
    }
}