# Delta Yatzy

Welcome to the Delta Yatzy project! This repository contains three main files for a web-based implementation of the classic dice game with additional features like optimal action suggestions and user evaluation tools.

## Files Overview

### 1. `index.html`
The core HTML structure of the Delta Yatzy application. It includes:
- A **scorecard** table for tracking categories, scores, and user actions.
- A **dice container** for managing and displaying the dice states.
- Sections for **optimal actions**, **histograms**, and **debugging information**.
- Integration with external libraries like Chart.js for visualizing data.
- References to `styles.css` and `app.js` for styling and functionality.

### 2. `styles.css`
The stylesheet defining the appearance of the application. It includes:
- Layout styles for the page and its elements (e.g., `container`, `right-column`).
- Specific styles for the **scorecard**, **dice elements**, and **buttons**.
- Highlights for game states (e.g., locked, reroll, scored categories).
- Styling for additional features like the histogram, loading overlay, and debug box.

### 3. `app.js`
The JavaScript logic powering the Delta Yatzy game. Key features include:
- **Player management**: Support for multiplayer with individual scorecards and dice states.
- **Dice interactions**: Dice rolling, reroll management, and UI updates for dice states.
- **Optimal action suggestions**: Communicates with APIs to recommend best actions based on the current game state.
- **Score evaluation**: Evaluates user actions and compares them with optimal actions, displaying differences.
- **Histograms**: Visual representation of expected scores using Chart.js.
- **State saving/restoration**: Saves and restores game progress locally to ensure continuity.

## Features
- Fully interactive **dice game UI**.
- **Optimal action suggestions** to guide players toward the best moves.
- Real-time **evaluation of user actions** and score tracking.
- Multiplayer support with individual player states.
- Integration with APIs for advanced gameplay analysis.

## Setup
1. Clone the repository.
2. Ensure you have a local server running to support API calls (e.g., `http://localhost:8080`).
3. Open `index.html` in a browser to start playing.

Enjoy your game of Delta Yatzy! 🎲