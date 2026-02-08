# ---------------------------------------------------------------
# Minimum required version of CMake
cmake_minimum_required(VERSION 3.22.1)

# ---------------------------------------------------------------
# Project name and version
project(Yatzy VERSION 1.0 LANGUAGES C)

# Specify the C standard
set(CMAKE_C_STANDARD 99)
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

# Library sources (everything except main entry points)
set(LIB_SOURCES
        ${SRC_DIR}/phase0_tables.c
        ${SRC_DIR}/widget_solver.c
        ${SRC_DIR}/api_computations.c
        ${SRC_DIR}/game_mechanics.c
        ${SRC_DIR}/dice_mechanics.c
        ${SRC_DIR}/webserver.c
        ${SRC_DIR}/context.c
        ${SRC_DIR}/simulations.c
        ${SRC_DIR}/storage.c
        ${SRC_DIR}/utilities.c
        ${SRC_DIR}/state_computation.c
)
message(STATUS "Source Directory: ${SRC_DIR}")

# Include project headers
include_directories(${INCLUDE_DIR})
message(STATUS "Include Directory: ${INCLUDE_DIR}")

# ---------------------------------------------------------------
# Main executable
add_executable(yatzy ${SRC_DIR}/yatzy.c ${LIB_SOURCES})

target_include_directories(yatzy PRIVATE ${MICROHTTPD_INCLUDE_DIR} ${JSON_C_INCLUDE_DIR})
target_link_libraries(yatzy PRIVATE ${MICROHTTPD_LIBRARY} ${JSON_C_LIBRARY})

# Add OpenMP flags explicitly
target_compile_options(yatzy PRIVATE -fopenmp)
target_link_libraries(yatzy PRIVATE gomp)

# ---------------------------------------------------------------
# Precompute executable
add_executable(yatzy_precompute ${SRC_DIR}/precompute.c ${LIB_SOURCES})
target_include_directories(yatzy_precompute PRIVATE ${MICROHTTPD_INCLUDE_DIR} ${JSON_C_INCLUDE_DIR})
target_link_libraries(yatzy_precompute PRIVATE ${MICROHTTPD_LIBRARY} ${JSON_C_LIBRARY})
target_compile_options(yatzy_precompute PRIVATE -fopenmp)
target_link_libraries(yatzy_precompute PRIVATE gomp)

# ---------------------------------------------------------------
# Test executables
enable_testing()

# Test include directory
set(TEST_DIR "${CMAKE_SOURCE_DIR}/tests")

# Helper macro: define a test that links against the yatzy library sources
macro(add_yatzy_test NAME)
    add_executable(${NAME} ${TEST_DIR}/${NAME}.c ${LIB_SOURCES})
    target_include_directories(${NAME} PRIVATE ${INCLUDE_DIR} ${TEST_DIR} ${MICROHTTPD_INCLUDE_DIR} ${JSON_C_INCLUDE_DIR})
    target_link_libraries(${NAME} PRIVATE ${MICROHTTPD_LIBRARY} ${JSON_C_LIBRARY} m)
    target_compile_options(${NAME} PRIVATE -fopenmp)
    target_link_libraries(${NAME} PRIVATE gomp)
    add_test(NAME ${NAME} COMMAND ${NAME})
endmacro()

add_yatzy_test(test_context)
add_yatzy_test(test_game_mechanics)
add_yatzy_test(test_dice_mechanics)
add_yatzy_test(test_phase0)
add_yatzy_test(test_widget)
add_yatzy_test(test_storage)

# Aggregate target to run all tests
add_custom_target(test_all COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
