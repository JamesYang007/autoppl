cmake_minimum_required(VERSION 3.10)
project(autoppl LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17) # set C++ standard to C++17

add_subdirectory(autoppl)
add_subdirectory(tests)
add_subdirectory(lib/googletest)