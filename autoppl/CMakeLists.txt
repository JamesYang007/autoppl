# Add this library as interface (header-only)
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME}
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Set C++17 standard for project target
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

#set(AUTOPPL_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
#file(GLOB_RECURSE AUTOPPL_SOURCE_FILES RELATIVE src LIST_DIRECTORIES false *.cpp)

#set(AUTOPPL_SOURCE_FILES ${AUTOPPL_SOURCE_DIR}/autoppl.cpp)
#set(AUTOPPL_HEADER_FILES ${AUTOPPL_INCLUDE_DIR}/autoppl.h)
