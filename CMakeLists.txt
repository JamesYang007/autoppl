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

# Add this library as interface (header-only)
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME}
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Set C++17 standard for project target
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

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

# TODO: add src dir if needed
#set(AUTOPPL_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
#file(GLOB_RECURSE AUTOPPL_SOURCE_FILES RELATIVE src LIST_DIRECTORIES false *.cpp)
#set(AUTOPPL_SOURCE_FILES ${AUTOPPL_SOURCE_DIR}/autoppl.cpp)
#set(AUTOPPL_HEADER_FILES ${AUTOPPL_INCLUDE_DIR}/autoppl.h)

# Add subdirectories
add_subdirectory(${PROJECT_SOURCE_DIR}/lib ${PROJECT_BINARY_DIR}/lib)
