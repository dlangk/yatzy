# Minimum required version of CMake
cmake_minimum_required(VERSION 3.22.1)

# Project name and version
project(Yatzy VERSION 1.0 LANGUAGES C)

# Specify the C standard
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Include project-specific directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Define source files
set(SRC_FILES
        src/main.c
        src/yatzy.c
        src/precompute_scores.c
        src/webserver.c
        src/file_utilities.c
)

# Define header files for clarity (optional)
set(HEADER_FILES
        include/constants.h
        include/webserver.h
        include/file_utilities.h
)

# Define the data directory as a compile-time definition
set(DATA_DIR ${CMAKE_SOURCE_DIR}/data)
add_compile_definitions(DATA_DIR="${DATA_DIR}")

# Locate and link libmicrohttpd
find_path(MICROHTTPD_INCLUDE_DIR NAMES microhttpd.h PATHS /opt/homebrew/include REQUIRED)
find_library(MICROHTTPD_LIBRARY NAMES microhttpd PATHS /opt/homebrew/lib REQUIRED)

# Locate and link json-c
find_path(JSON_C_INCLUDE_DIR NAMES json-c/json.h PATHS /opt/homebrew/include REQUIRED)
find_library(JSON_C_LIBRARY NAMES json-c PATHS /opt/homebrew/lib REQUIRED)

# Add the executable
add_executable(yatzy ${SRC_FILES} ${HEADER_FILES})

# Ensure headers and external libraries are included
target_include_directories(yatzy PRIVATE ${MICROHTTPD_INCLUDE_DIR} ${JSON_C_INCLUDE_DIR})
target_link_libraries(yatzy PRIVATE ${MICROHTTPD_LIBRARY} ${JSON_C_LIBRARY})

# Add OpenMP flags explicitly
target_compile_options(yatzy PRIVATE -fopenmp)
target_link_libraries(yatzy PRIVATE gomp)

# Installation (optional, adds binary to 'bin' directory)
install(TARGETS yatzy RUNTIME DESTINATION bin)

# Custom run target for convenience
add_custom_target(run
        COMMAND yatzy
        DEPENDS yatzy
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)