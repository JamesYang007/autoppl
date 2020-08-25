cmake_minimum_required(VERSION 3.7)
project("AutoPPL"
        VERSION 1.1
        DESCRIPTION "C++ template library for probabilistic programming."
        LANGUAGES C CXX)

option(AUTOPPL_ENABLE_TEST "Enable unit tests to be built." ON)
option(AUTOPPL_ENABLE_BENCHMARK "Enable benchmarks to be built." OFF)
option(AUTOPPL_ENABLE_TEST_COVERAGE "Build with test coverage (AUTOPPL_ENABLE_TEST must be ON)" OFF)
option(AUTOPPL_ENABLE_EXAMPLE "Enable compilation of examples." OFF)

# This is to make this library portable to other machines.
# This will be used for install.
include(GNUInstallDirs)

# Add AutoPPL header-only library to variable
set(AUTOPPL_LIBS ${PROJECT_NAME})

# Find Eigen3
find_package(Eigen3 3.3 CONFIG REQUIRED NO_MODULE
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/FastAD/libs/eigen-3.3.7/build/share)
    set(AUTOPPL_LIBS ${AUTOPPL_LIBS} Eigen3::Eigen)
message(STATUS "Eigen3 found at ${EIGEN3_INCLUDE_DIR}")

# Find FastAD
find_package(FastAD CONFIG REQUIRED
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/FastAD/build)
if (FastAD_FOUND)
    message(STATUS "Found FastAD config at: ${FastAD_DIR}")
    set(AUTOPPL_LIBS ${AUTOPPL_LIBS} FastAD::FastAD)
endif()

# Add this library as interface (header-only)
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME}
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

# Set install destinations
install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Create AutoPPLConfigVersion.cmake which contains current project version
# This is supposed to help with (major) version compatibility.
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY SameMajorVersion)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION
    ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(FILES
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})

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
    add_subdirectory(${PROJECT_SOURCE_DIR}/example ${PROJECT_BINARY_DIR}/example)
endif()
