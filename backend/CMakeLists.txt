# ---------------------------------------------------------------
# Minimum required version of CMake
cmake_minimum_required(VERSION 3.22.1)

# ---------------------------------------------------------------
# Project name and version
project(Yatzy VERSION 1.0 LANGUAGES C)

# Specify the C standard
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------
# Include directories for Homebrew-installed libraries
include_directories(/opt/homebrew/include)
link_directories(/opt/homebrew/lib)

# Locate and link libmicrohttpd
find_path(MICROHTTPD_INCLUDE_DIR NAMES microhttpd.h PATHS /opt/homebrew/include REQUIRED)
find_library(MICROHTTPD_LIBRARY NAMES microhttpd PATHS /opt/homebrew/lib REQUIRED)

# Locate and link json-c
find_path(JSON_C_INCLUDE_DIR NAMES json-c/json.h PATHS /opt/homebrew/include REQUIRED)
find_library(JSON_C_LIBRARY NAMES json-c PATHS /opt/homebrew/lib REQUIRED)

# ---------------------------------------------------------------
# Set up paths for source files and headers
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

# Define the source files
set(SOURCES
        ${SRC_DIR}/yatzy.c
        ${SRC_DIR}/computations.c
        ${SRC_DIR}/game_mechanics.c
        ${SRC_DIR}/dice_mechanics.c
        ${SRC_DIR}/webserver.c
        ${SRC_DIR}/context.c
        ${SRC_DIR}/simulations.c
        src/storage.c
        include/storage.h
        src/simulations.c
)
message(STATUS "Source Directory: ${SRC_DIR}")

# Include project headers
include_directories(${INCLUDE_DIR})
message(STATUS "Include Directory: ${INCLUDE_DIR}")

# ---------------------------------------------------------------
# Define the executable
add_executable(yatzy ${SOURCES})

# Include and link packages
target_include_directories(yatzy PRIVATE ${MICROHTTPD_INCLUDE_DIR} ${JSON_C_INCLUDE_DIR})
target_link_libraries(yatzy PRIVATE ${MICROHTTPD_LIBRARY} ${JSON_C_LIBRARY})

# Add OpenMP flags explicitly
target_compile_options(yatzy PRIVATE -fopenmp)
target_link_libraries(yatzy PRIVATE gomp)