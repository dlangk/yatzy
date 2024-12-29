# Yatzy Backend

This project implements a backend for managing a Yatzy game with an API server. It includes game logic, state precomputation, file utilities, and a webserver to interact with the game.

## Features

### 1. **Game Logic (`yatzy.c`)**
- Implements core Yatzy game functionality, including:
    - Rolling and rerolling dice.
    - Calculating scores for various Yatzy categories.
    - Evaluating optimal actions based on the current state.
    - Simulating a single Yatzy game for testing and validation.

### 2. **State Precomputation (`precompute_scores.c`)**
- Precomputes essential game data to optimize runtime performance:
    - Precomputes category scores for all dice combinations.
    - Precomputes transition probabilities for reroll scenarios.
    - Precomputes expected state values for various game states.
- Saves and loads state values to/from binary files for efficiency.

### 3. **File Utilities (`file_utilities.c`)**
- Handles file-related operations:
    - Parses CSV files and extracts data.
    - Saves and loads precomputed game state values to binary files.
    - Checks file existence to manage data persistence.

### 4. **API Server (`webserver.c`)**
- Provides a RESTful API for interacting with the game:
    - Endpoints for evaluating actions, retrieving scores, and more.
    - Handles HTTP GET and POST requests using `libmicrohttpd`.
    - Adds CORS headers for cross-origin requests.
- Runs on a polling thread to handle multiple connections.

### 5. **Main Entry Point (`main.c`)**
- Initializes the game backend and starts the API server.
- Gracefully handles shutdown signals (e.g., `Ctrl+C`) to clean up resources.
- Keeps the server running until interrupted.

## Build and Run

### Prerequisites
- **C Compiler:** GCC 14 or compatible.
- **Libraries:**
    - [libmicrohttpd](https://www.gnu.org/software/libmicrohttpd/)
    - [json-c](https://github.com/json-c/json-c)

### Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=gcc-14
make
```
