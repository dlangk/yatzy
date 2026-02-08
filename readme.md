# Delta Yatzy Project

Delta Yatzy is a web-based implementation of the classic dice game Yatzy, designed to provide an engaging and interactive experience. This project includes both a **backend API** for game logic and state management and a **frontend** for an intuitive user interface. Together, these components offer features such as optimal action suggestions, score evaluation, and multiplayer support.

---

## Features

### **Backend**
The backend powers the game with precomputed data, efficient file management, and a RESTful API.

#### Key Features:
- **Game Logic**: Implements core Yatzy gameplay, including rolling dice, calculating scores, and evaluating optimal actions.
- **State Precomputation**: Optimizes runtime performance with precomputed category scores and transition probabilities. Features real-time progress tracking and memory-mapped I/O for efficient computation.
- **File Utilities**: Manages game state persistence with consolidated binary file format (all_states.bin) and memory-mapped I/O for fast loading.
- **API Server**: Provides endpoints for evaluating actions, retrieving scores, and more, built with `libmicrohttpd`.

#### Technologies:
- **C Programming Language**
- **libmicrohttpd** for HTTP server
- **json-c** for JSON handling

#### Build and Run:
```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=gcc-14
make

# Run the server
./yatzy´´´

### **Frontend**

The frontend is an interactive web application that communicates with the backend to provide a seamless user experience.

#### Key Features:
- **Interactive UI**: Features a dynamic scorecard, dice container, and histogram visualizations.
- **Optimal Suggestions**: Guides players with API-driven optimal actions and comparisons to user actions.
- **Multiplayer Support**: Tracks individual player states and scores for multiplayer games.
- **Visualizations**: Uses Chart.js for real-time histograms and score distributions.

#### Technologies:
- **HTML/CSS/JavaScript**
- **Chart.js** for data visualization
- Integration with the backend API.

#### Setup:
1. Clone the repository.
2. Start the backend server (`./yatzy`).
3. Open `index.html` in a browser with a local server.

---

## API Example

### Endpoint: `/state_value`

**Request:**
```bash
curl http://localhost:8080/state_value?upper_score=50&scored_categories=3

## Project Structure

### Backend
- **Core Logic:** `yatzy.c` - Entry point and context initialization
- **State Computation:** `state_computation.c` - Dynamic programming engine with progress tracking
- **Game Mechanics:** `game_mechanics.c` - Core Yatzy rules and scoring
- **API Server:** `webserver.c` - RESTful API endpoints
- **Computations:** `computations.c` - Expected value calculations

### Frontend
- **HTML:** `index.html` (UI structure)
- **CSS:** `styles.css` (UI styling)
- **JavaScript:** `interface.js` (game logic and backend communication)