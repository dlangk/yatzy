# Yatzy Backend

This project implements a backend for managing a Yatzy game with an API server. It includes game logic, state
precomputation, file utilities, and a webserver to interact with the game.

## Features

### 1. **Game Logic (`yatzy.c`)**

- Implements core Yatzy game functionality, including:
    - Rolling and rerolling dice.
    - Calculating scores for various Yatzy categories.
    - Evaluating optimal actions based on the current state.
    - Simulating a single Yatzy game for testing and validation.

### 2. **State Computation (`state_computation.c`)**

- Implements dynamic programming engine for computing optimal game strategies:
    - Uses backward induction to compute expected values for all game states.
    - Features real-time progress tracking with ETA calculations.
    - Utilizes memory-mapped I/O for fast file operations.
    - Supports parallel computation using OpenMP.
    - Automatically saves progress to resume interrupted computations.
    - Creates consolidated binary file (`all_states.bin`) for fast loading.

### 3. **File Utilities**

- Binary file format with magic number and versioning for compatibility.
- Consolidated state file (`data/all_states.bin`) contains all precomputed values.
- Memory-mapped I/O support for efficient loading of large state files.
- Automatic fallback to per-level files if consolidated file not found.

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

# Yatzy Backend Setup

This project requires several dependencies to be installed before you can build and run the backend.

## Dependencies

1. **CMake** (version >= 3.30.5)
2. **libmicrohttpd** (for the webserver)
3. **json-c** (for JSON handling)
4. **gcc** and **OpenMP** (for multithreading)

---

## Installation Instructions

### Ubuntu

    sudo apt update
    sudo apt install cmake libmicrohttpd-dev libjson-c-dev gcc build-essential -y

### MacOS

    brew install cmake libmicrohttpd json-c gcc

---

## Build Instructions

1. Navigate to the `backend` folder:

       cd backend

2. Create a build directory and compile:

       mkdir build
       cd build
       cmake ..
       make

3. Run the backend server:

       ./yatzy

---

## Troubleshooting

- **Check CMake version**:

       cmake --version

- **Ensure headers/libraries are found**:

       find /usr/include -name microhttpd.h
       find /usr/include -name json.h

- **Verify compiler installation**:

       gcc --version