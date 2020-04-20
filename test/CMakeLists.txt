# Configure test coverage
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    message(STATUS "Building test suite with coverage information")
    add_compile_options(--coverage -O0 -fno-inline -fno-inline-small-functions
                        -fno-default-inline)
endif()

######################################################
# Util Test
######################################################

add_executable(util_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/util/concept_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/dist_expr_traits_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/var_expr_traits_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/var_traits_unittest.cpp
    )
target_compile_options(util_unittest PRIVATE -g -Wall -Werror -Wextra)
target_include_directories(util_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(util_unittest gcov)
endif()
target_link_libraries(util_unittest gtest_main pthread ${PROJECT_NAME})
add_test(util_unittest util_unittest)

######################################################
# Variable Expression Test
######################################################

add_executable(var_expr_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/variable/variable_viewer_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/variable/constant_unittest.cpp
    )
target_compile_options(var_expr_unittest PRIVATE -g -Wall -Werror -Wextra)
target_include_directories(var_expr_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(var_expr_unittest gcov)
endif()
target_link_libraries(var_expr_unittest gtest_main pthread ${PROJECT_NAME})
add_test(var_expr_unittest var_expr_unittest)

######################################################
# Distribution Expression Test
######################################################

add_executable(dist_expr_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/distribution/bernoulli_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/distribution/density_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/distribution/normal_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/distribution/uniform_unittest.cpp
    )
target_compile_options(dist_expr_unittest PRIVATE -g -Wall -Werror -Wextra)
target_include_directories(dist_expr_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(dist_expr_unittest gcov)
endif()
target_link_libraries(dist_expr_unittest gtest_main pthread ${PROJECT_NAME})
add_test(dist_expr_unittest dist_expr_unittest)

######################################################
# Model Expression Test
######################################################

add_executable(model_expr_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/model/model_unittest.cpp
    )
target_compile_options(model_expr_unittest PRIVATE -g -Wall -Werror -Wextra)
target_include_directories(model_expr_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(model_expr_unittest gcov)
endif()
target_link_libraries(model_expr_unittest gtest_main pthread ${PROJECT_NAME})
add_test(model_expr_unittest model_expr_unittest)

######################################################
# Algorithm Test
######################################################

add_executable(algorithm_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithm/mh_unittest.cpp
    )
target_compile_options(algorithm_unittest PRIVATE -g -Wall -Werror -Wextra)
target_include_directories(algorithm_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(algorithm_unittest gcov)
endif()
target_link_libraries(algorithm_unittest gtest_main pthread ${PROJECT_NAME})
add_test(algorithm_unittest algorithm_unittest)

######################################################
# Expression Builder Test
######################################################

add_executable(expr_builder_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/expr_builder_unittest.cpp
    )
target_compile_options(expr_builder_unittest PRIVATE -g -Wall -Werror -Wextra)
target_include_directories(expr_builder_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(expr_builder_unittest gcov)
endif()
target_link_libraries(expr_builder_unittest gtest_main pthread ${PROJECT_NAME})
add_test(expr_builder_unittest expr_builder_unittest)
