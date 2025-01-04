const elements = {
    evaluateButtons: document.querySelectorAll(".evaluate-category"),
    histogramCanvas: document.getElementById("histogramCanvas"), // Match usage in renderHistogram
    playerSelect: document.getElementById("playerSelect"),
    addPlayerButton: document.getElementById("addPlayerButton"),
    scorecardTable: document.getElementById("scorecard"),
    diceContainer: document.getElementById("diceContainer"),
    rerollsRemaining: document.getElementById("rerollsRemaining"),
    rerollButton: document.getElementById("rerollDiceButton"),
    randomizeButton: document.getElementById("randomizeDiceButton"),
    deltaBox: document.getElementById("deltaBox"),
    sortedDiceContainer: document.getElementById("sortedDice"),
    optimalActionButton: document.getElementById("toggleOptimalActionButton"),
    instructionsButton: document.getElementById("toggleInstructionsButton"),
    userEvaluation: document.getElementById("userEvaluation"),
    userActionTextArea: document.getElementById("rawUserEvaluation"),
    optimalActionBox: document.getElementById("optimalActionBox"),
    instructionsBox: document.getElementById("instructionsBox"),
    debugButton: document.getElementById("toggleDebugButton"),
    debugBox: document.getElementById("debugBox"),
};

export default elements;