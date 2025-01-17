import {TOTAL_DICE, YAHTZEE_CATEGORIES} from "../utils/constants.js";
import {runRefreshers} from "../utils/refreshers.js";
import {GAME_STATE_LOCALSTORAGE, BONUS_SCORE, BONUS_THRESHOLD} from "../utils/constants.js";

const gameState = {
    players: [],
    activePlayerIndex: 0,
    scorecards: [],
    uiState: {
        showOptimalAction: false,
        showDebugInfo: false,
        showInstructions: false,
    },

    createDefaultPlayer(name) {
        return {
            name,
            diceState: Array.from({length: TOTAL_DICE}, () => ({
                value: Math.floor(Math.random() * 6) + 1,
                isLocked: true,
            })),
            optimalRerollMask: [],
            upperScore: 0,
            totalScore: 0,
            bonus: 0,
            rerollsRemaining: 2,
            userAction: null,
            userActionResult: null,
            optimalActionResult: null,
            optimalCategoryId: null,
            histogramData: null,
        };
    },

    // Mutating Methods (Ensure saveState() is called)
    initializePlayerScorecard(playerIndex) {
        this.scorecards[playerIndex] = YAHTZEE_CATEGORIES.map((category, index) => ({
            id: index,
            name: category,
            score: 0,
            isScored: false,
            isValid: false,
        }));
        this.saveState();
    },

    updateCategoryValidity(categoryId, isValid) {
        const activePlayer = this.getActivePlayer();
        if (!activePlayer) return;

        const category = activePlayer.scorecard.find(cat => cat.id === categoryId);
        if (category) {
            category.isValid = isValid;
        }

        this.saveState();
    },

    toggleOptimalAction() {
        this.uiState.showOptimalAction = !this.uiState.showOptimalAction;
        this.saveState();
    },

    toggleInstructions() {
        this.uiState.showInstructions = !this.uiState.showInstructions;
        this.saveState();
    },

    initializePlayers() {
        this.players = [];
        this.addPlayer();
        this.activePlayerIndex = 0;
        this.saveState();
    },

    addPlayer() {
        this.players.push(this.createDefaultPlayer(`Player ${this.players.length + 1}`));
        this.initializePlayerScorecard(this.players.length - 1);
        this.saveState();
    },

    getActivePlayerDiceState() {
        return this.getActivePlayer()?.diceState || [];
    },

    async updateDieValue(index, newValue) {
        const diceState = this.getActivePlayerDiceState();
        if (diceState[index]) {
            diceState[index].value = newValue;
            await runRefreshers();
            this.saveState();
        }
    },

    async toggleDieLockState(index) {
        const diceState = this.getActivePlayerDiceState();
        if (diceState[index]) {
            diceState[index].isLocked = !diceState[index].isLocked;
            this.saveState();
        }
    },

    async rerollDice() {
        const activePlayer = this.getActivePlayer();
        if (!activePlayer || activePlayer.rerollsRemaining <= 0) return;

        activePlayer.diceState.forEach(die => {
            if (!die.isLocked) {
                die.value = Math.floor(Math.random() * 6) + 1;
            }
        });

        activePlayer.rerollsRemaining -= 1;
        this.saveState();
        await runRefreshers(); // Refresh available categories after reroll
    },

    async randomizeDice() {
        const activePlayer = this.getActivePlayer();
        if (!activePlayer) return;

        activePlayer.rerollsRemaining = 2;
        activePlayer.diceState = activePlayer.diceState.map(() => ({
            value: Math.floor(Math.random() * 6) + 1,
            isLocked: true,
        }));
        this.saveState();
        await runRefreshers(); // Refresh available categories after randomization
    },

    setActivePlayerIndex(index) {
        if (index >= 0 && index < this.players.length) {
            this.activePlayerIndex = index;
            this.saveState();
        } else {
            console.warn(`Invalid player index: ${index}`);
        }
    },

    // Utility Methods (No saveState() needed)

    getActivePlayer() {
        return this.players[this.activePlayerIndex];
    },

    getActivePlayerScorecard() {
        return this.scorecards[this.activePlayerIndex];
    },

    getUpperScore(playerIndex = this.activePlayerIndex) {
        const scorecard = this.scorecards[playerIndex];
        return scorecard
            .filter(category => category.id <= 5 && category.isScored)
            .reduce((sum, category) => sum + category.score, 0);
    },

    getTotalScore(playerIndex = this.activePlayerIndex) {
        const scorecard = this.scorecards[playerIndex];
        return scorecard
            .filter(category => category.isScored)
            .reduce((sum, category) => sum + category.score, 0);
    },

    getBonus(playerIndex = this.activePlayerIndex) {
        return this.getUpperScore(playerIndex) >= BONUS_THRESHOLD ? BONUS_SCORE : 0;
    },

    getScoredCategoriesBitmask(playerIndex = this.activePlayerIndex) {
        const scorecard = this.scorecards[playerIndex];
        return scorecard.reduce((bitmask, category) =>
            category.isScored ? bitmask | (1 << category.id) : bitmask, 0);
    },

    getEvaluationPayload(categoryId) {
        const activePlayer = this.getActivePlayer();
        if (!activePlayer) return null;

        return {
            upper_score: this.getUpperScore(),
            scored_categories: this.getScoredCategoriesBitmask(),
            dice: activePlayer.diceState.map(die => die.value),
            rerolls_remaining: activePlayer.rerollsRemaining,
            user_action: {
                best_category: {
                    id: categoryId,
                    name: YAHTZEE_CATEGORIES[categoryId],
                },
            },
        };
    },

    saveState() {
        console.log("Saving game state...");
        localStorage.setItem(GAME_STATE_LOCALSTORAGE, JSON.stringify(this));
    },

    loadState() {
        const savedState = JSON.parse(localStorage.getItem(GAME_STATE_LOCALSTORAGE));
        if (!savedState) return;

        const hydratedState = {
            players: savedState.players || [],
            activePlayerIndex: savedState.activePlayerIndex || 0,
            scorecards: savedState.scorecards || [],
            uiState: {
                showOptimalAction: savedState.uiState?.showOptimalAction || false,
                showDebugInfo: savedState.uiState?.showDebugInfo || false,
                showInstructions: savedState.uiState?.showInstructions || false,
            },
        };

        Object.assign(this, hydratedState);

        this.players.forEach((_, index) => {
            if (!this.scorecards[index]) {
                this.initializePlayerScorecard(index);
            }
        });
    },

    resetGame() {
        console.log("Resetting game state...");
        localStorage.removeItem(GAME_STATE_LOCALSTORAGE);
        this.initializePlayers();
        this.randomizeDice().then(() => {
            this.saveState();
        });
    }
};

export default gameState;

export function loadOrCreateGameState() {
    if (localStorage.getItem(GAME_STATE_LOCALSTORAGE)) {
        console.log("Loading existing game state...");
        gameState.loadState();
    } else {
        console.log("Creating new game state...");
        gameState.initializePlayers();
        gameState.randomizeDice().then(() => {
            gameState.saveState();
        })
    }
    window.gameState = gameState;
    runRefreshers().then(() => console.log("Finished creating game state."));
}