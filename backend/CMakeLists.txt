# Minimum required version of CMake
cmake_minimum_required(VERSION 3.22.1)

# Project name and version
project(Yatzy VERSION 1.0 LANGUAGES C)

# Specify the C standard
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Include directories for Homebrew-installed libraries
include_directories(/opt/homebrew/include)
link_directories(/opt/homebrew/lib)

# Locate and link libmicrohttpd
find_path(MICROHTTPD_INCLUDE_DIR NAMES microhttpd.h PATHS /opt/homebrew/include REQUIRED)
find_library(MICROHTTPD_LIBRARY NAMES microhttpd PATHS /opt/homebrew/lib REQUIRED)

# Locate and link json-c
find_path(JSON_C_INCLUDE_DIR NAMES json-c/json.h PATHS /opt/homebrew/include REQUIRED)
find_library(JSON_C_LIBRARY NAMES json-c PATHS /opt/homebrew/lib REQUIRED)

# Define the executable
add_executable(yatzy yatzy.c)

# Ensure headers are included
target_include_directories(yatzy PRIVATE ${MICROHTTPD_INCLUDE_DIR} ${JSON_C_INCLUDE_DIR})

# Link libraries to the target
target_link_libraries(yatzy PRIVATE ${MICROHTTPD_LIBRARY} ${JSON_C_LIBRARY})

# Add OpenMP flags explicitly
target_compile_options(yatzy PRIVATE -fopenmp)
target_link_libraries(yatzy PRIVATE gomp)