cmake_minimum_required(VERSION 3.10)
project("autoppl"
        VERSION 0.8
        DESCRIPTION "C++ template library for probabilistic programming."
        LANGUAGES C CXX)

# This will perform memcheck
include(CTest)

# enables testing
enable_testing()

# Automate the choosing of config
# if CMAKE_BUILD_TYPE not defined
if (NOT CMAKE_BUILD_TYPE)
    # if binary directory ends with "release", use release mode
    if (${PROJECT_BINARY_DIR} MATCHES "release$")
        set(CMAKE_BUILD_TYPE RELEASE)
    # otherwise, use debug mode
    else()
        set(CMAKE_BUILD_TYPE DEBUG)
    endif()
endif()
message(STATUS "Compiling in ${CMAKE_BUILD_TYPE} mode")

# Set directories for third-party libraries
# tests in other subdirectories require access to these as well
set(GTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/benchmark/googletest/googletest)
set(GBENCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/benchmark)

# TODO: don't think this is necessary since it will be header-only
add_subdirectory(${PROJECT_SOURCE_DIR}/autoppl ${PROJECT_BINARY_DIR}/autoppl)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests ${PROJECT_BINARY_DIR}/tests)
add_subdirectory(${PROJECT_SOURCE_DIR}/libs ${PROJECT_BINARY_DIR}/libs)
