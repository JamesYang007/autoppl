set(BINARY ${CMAKE_PROJECT_NAME})

set(AUTOPPL_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
set(AUTOPPL_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)

file(GLOB AUTOPPL_SOURCE_FILES src/*.cpp)
file(GLOB AUTOPPL_HEADER_FILES include/*.h)

add_library(${BINARY}_lib ${AUTOPPL_SOURCE_FILES} ${AUTOPPL_HEADER_FILES})
target_include_directories(${BINARY}_lib PUBLIC ${AUTOPPL_INCLUDE_DIR})

add_executable(${BINARY} main.cpp)
target_link_libraries(${BINARY} PUBLIC autoppl_lib)
