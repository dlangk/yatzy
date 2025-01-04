import {attachEventHandlers} from "./modules/utils/eventHandlers.js";
import {renderUI} from "./modules/utils/uiBuilders.js";
import {loadOrCreateGameState} from "./modules/game/gameState.js";

function init() {
    console.log("Initializing...");
    loadOrCreateGameState();
    renderUI();
    attachEventHandlers();
}

document.addEventListener("DOMContentLoaded", () => {
    init();
    document.body.classList.add("loaded");
});