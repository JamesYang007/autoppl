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
    )
target_compile_options(expression_unittest PRIVATE -g -Wall -Werror -Wextra)
target_include_directories(expression_unittest PRIVATE ${GTEST_DIR}/include)
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(expression_unittest gcov)
endif()
target_link_libraries(expression_unittest gtest_main pthread ${PROJECT_NAME})
add_test(expression_unittest expression_unittest)

######################################################
# Distribution Test
######################################################

add_executable(distribution_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/distribution/uniform_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/distribution/normal_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/distribution/bernoulli_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/distribution/discrete_unittest.cpp

    )
target_compile_options(distribution_unittest PRIVATE -g -Wall -Werror -Wextra)
target_include_directories(distribution_unittest PRIVATE ${GTEST_DIR}/include)
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(distribution_unittest gcov)
endif()
target_link_libraries(distribution_unittest gtest_main pthread ${PROJECT_NAME})
add_test(distribution_unittest distribution_unittest)

######################################################
# Algorithm Test
######################################################

add_executable(algorithm_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithm/mh_unittest.cpp
    )
target_compile_options(algorithm_unittest PRIVATE -g -Wall -Werror -Wextra)
target_include_directories(algorithm_unittest PRIVATE ${GTEST_DIR}/include)
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(algorithm_unittest gcov)
endif()
target_link_libraries(algorithm_unittest gtest_main pthread ${PROJECT_NAME})
add_test(algorithm_unittest algorithm_unittest)
