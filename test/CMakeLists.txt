# Configure test coverage
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    message(STATUS "Building test suite with coverage information")
    add_compile_options(--coverage -O0 -fno-inline -fno-inline-small-functions
                        -fno-default-inline)
endif()

######################################################
# Expression Test
######################################################

add_executable(expression_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/model_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/uniform_unittest.cpp
    )
target_compile_options(expression_unittest PRIVATE -g -Wall -Werror -Wextra)
target_include_directories(expression_unittest PRIVATE ${GTEST_DIR}/include)
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(expression_unittest gcov)
endif()
target_link_libraries(expression_unittest gtest_main pthread ${PROJECT_NAME})
add_test(expression_unittest expression_unittest)
