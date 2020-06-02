cmake_minimum_required(VERSION 3.7)
project("autoppl"
        VERSION 0.8
        DESCRIPTION "C++ template library for probabilistic programming."
        LANGUAGES C CXX)

option(AUTOPPL_ENABLE_TEST "Enable unit tests to be built." ON)
# TODO: later when we make benchmarks, this should be ON
option(AUTOPPL_ENABLE_BENCHMARK "Enable benchmarks to be built." OFF)
option(AUTOPPL_ENABLE_TEST_COVERAGE "Build with test coverage (AUTOPPL_ENABLE_TEST must be ON)" OFF)
option(AUTOPPL_ENABLE_EXAMPLE "Enable compilation of examples." OFF)

# Automate the choosing of config
if (NOT CMAKE_BUILD_TYPE)
    # if binary directory ends with "release", use release mode
    if (${PROJECT_BINARY_DIR} MATCHES "release$")
        set(CMAKE_BUILD_TYPE RELEASE)
        add_compile_options(-O3)
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
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

# Add AutoPPL header-only library to variable
set(AUTOPPL_LIBS ${PROJECT_NAME})

# Find Armadillo C++
find_package(Armadillo CONFIG REQUIRED
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/armadillo)
if (Armadillo_FOUND)
    message(STATUS "Found Armadillo config at: ${Armadillo_DIR}")
    add_compile_options(-DARMA_DONT_USE_WRAPPER)
    set(AUTOPPL_INCLUDE_DIRS ${AUTOPPL_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})
    set(AUTOPPL_LIBS ${AUTOPPL_LIBS} ${ARMADILLO_LIBRARIES})
endif()

# Find FastAD
find_package(FastAD CONFIG REQUIRED
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/FastAD/build)
if (FastAD_FOUND)
    message(STATUS "Found FastAD config at: ${FastAD_DIR}")
    set(AUTOPPL_LIBS ${AUTOPPL_LIBS} FastAD::FastAD)
endif()

# Set directory for GoogleTest
set(GTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/benchmark/googletest/googletest)
set(GBENCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/benchmark)

# Add lib subdirectory
add_subdirectory(${PROJECT_SOURCE_DIR}/lib ${PROJECT_BINARY_DIR}/lib)

# Configure tests
if (AUTOPPL_ENABLE_TEST)
    include(CTest) # enable memcheck
    enable_testing()
    add_subdirectory(${PROJECT_SOURCE_DIR}/test ${PROJECT_BINARY_DIR}/test)
endif()

# Configure benchmarking
if (AUTOPPL_ENABLE_BENCHMARK)
	add_subdirectory(${PROJECT_SOURCE_DIR}/benchmark ${PROJECT_BINARY_DIR}/benchmark)
endif()

# Compile examples if enabled
if (AUTOPPL_ENABLE_EXAMPLE)
    add_subdirectory(${PROJECT_SOURCE_DIR}/docs/example ${PROJECT_BINARY_DIR}/example)
endif()
