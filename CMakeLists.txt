cmake_minimum_required(VERSION 3.7)
project("autoppl"
        VERSION 0.8
        DESCRIPTION "C++ template library for probabilistic programming."
        LANGUAGES C CXX)

option(AUTOPPL_ENABLE_TEST "Enable unit tests to be built." ON)
# TODO: later when we make benchmarks, this should be ON
option(AUTOPPL_ENABLE_BENCHMARK "Enable benchmarks to be built." OFF)
option(AUTOPPL_ENABLE_TEST_COVERAGE "Build with test coverage (AUTOPPL_ENABLE_TEST must be ON)" OFF)

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

# Configure tests
if (AUTOPPL_ENABLE_TEST)
    include(CTest) # enable memcheck
    enable_testing()

    # Set directories for third-party libraries
    # tests in other subdirectories require access to these as well
    set(GTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/benchmark/googletest/googletest)
    set(GBENCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/benchmark)

    add_subdirectory(${PROJECT_SOURCE_DIR}/test ${PROJECT_BINARY_DIR}/test)
endif()

# Add subdirectories
add_subdirectory(${PROJECT_SOURCE_DIR}/autoppl ${PROJECT_BINARY_DIR}/autoppl)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib ${PROJECT_BINARY_DIR}/lib)
