cmake_minimum_required(VERSION 3.10)
project(autoppl LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17) # set C++ standard to C++17
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(autoppl)

option(PACKAGE_TESTS "Build the tests" ON)
if(PACKAGE_TESTS)
    enable_testing()
    add_subdirectory(lib/googletest)
    add_subdirectory(tests)
endif()
