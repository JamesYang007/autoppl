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

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_options(util_unittest PRIVATE -g -Wall)
else()
	target_compile_options(util_unittest PRIVATE -g -Wall -Werror -Wextra)
endif()

target_include_directories(util_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AUTOPPL_INCLUDE_DIRS}
    )
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(util_unittest gcov)
endif()

target_link_libraries(util_unittest autoppl_gtest_main ${AUTOPPL_LIBS})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_link_libraries(util_unittest pthread)
endif()

add_test(util_unittest util_unittest)

######################################################
# Sample Test
######################################################

add_executable(sample_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/samples/dist_sample_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/samples/model_sample_unittest.cpp
    )

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_options(sample_unittest PRIVATE -g -Wall)
else()
	target_compile_options(sample_unittest PRIVATE -g -Wall -Werror -Wextra)
endif()

target_include_directories(sample_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AUTOPPL_INCLUDE_DIRS}
    )
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(sample_unittest gcov)
endif()

target_link_libraries(sample_unittest autoppl_gtest_main ${AUTOPPL_LIBS})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_link_libraries(sample_unittest pthread)
endif()

add_test(sample_unittest sample_unittest)

######################################################
# Variable Test
######################################################

add_executable(var_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/variable/variable_viewer_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/variable/param_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/variable/data_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/variable/constant_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/variable/binop_unittest.cpp
    )

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_options(var_unittest PRIVATE -g -Wall)
else()
	target_compile_options(var_unittest PRIVATE -g -Wall -Werror -Wextra)
endif()

target_include_directories(var_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AUTOPPL_INCLUDE_DIRS}
    )
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(var_unittest gcov)
endif()

target_link_libraries(var_unittest autoppl_gtest_main ${AUTOPPL_LIBS})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_link_libraries(var_unittest pthread)
endif()

add_test(var_unittest var_unittest)

######################################################
# Distribution Expression Test
######################################################

add_executable(dist_expr_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/distribution/bernoulli_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/distribution/normal_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/distribution/uniform_unittest.cpp
    )

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_options(dist_expr_unittest PRIVATE -g -Wall)
else()
	target_compile_options(dist_expr_unittest PRIVATE -g -Wall -Werror -Wextra)
endif()

target_include_directories(dist_expr_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AUTOPPL_INCLUDE_DIRS}
    )
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(dist_expr_unittest gcov)
endif()

target_link_libraries(dist_expr_unittest autoppl_gtest_main ${AUTOPPL_LIBS})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_link_libraries(dist_expr_unittest pthread)
endif()

add_test(dist_expr_unittest dist_expr_unittest)

######################################################
# Model Expression Test
######################################################

add_executable(model_expr_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/model/model_unittest.cpp
    )

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_options(model_expr_unittest PRIVATE -g -Wall)
else()
	target_compile_options(model_expr_unittest PRIVATE -g -Wall -Werror -Wextra)
endif()

target_include_directories(model_expr_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AUTOPPL_INCLUDE_DIRS}
    )
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(model_expr_unittest gcov)
endif()

target_link_libraries(model_expr_unittest autoppl_gtest_main ${AUTOPPL_LIBS})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_link_libraries(model_expr_unittest pthread)
endif()

add_test(model_expr_unittest model_expr_unittest)

######################################################
# Math Test
######################################################

add_executable(math_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/math/welford_unittest.cpp
    )

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_options(math_unittest PRIVATE -g -Wall)
else()
	target_compile_options(math_unittest PRIVATE -g -Wall -Werror -Wextra)
endif()

target_include_directories(math_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AUTOPPL_INCLUDE_DIRS}
    )
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(math_unittest gcov)
endif()

target_link_libraries(math_unittest autoppl_gtest_main ${AUTOPPL_LIBS})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_link_libraries(math_unittest pthread)
endif()

add_test(math_unittest math_unittest)

######################################################
# MCMC Test
######################################################

add_executable(mcmc_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/mcmc/mh_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mcmc/mh_regression_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mcmc/sampler_tools_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mcmc/hmc/var_adapter_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mcmc/hmc/nuts/nuts_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mcmc/hmc/hamiltonian_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mcmc/hmc/leapfrog_unittest.cpp
    )

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_options(mcmc_unittest PRIVATE -g -Wall)
else()
	target_compile_options(mcmc_unittest PRIVATE -g -Wall -Werror -Wextra)
endif()

target_include_directories(mcmc_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AUTOPPL_INCLUDE_DIRS}
    )
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(mcmc_unittest gcov)
endif()

target_link_libraries(mcmc_unittest autoppl_gtest_main ${AUTOPPL_LIBS})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_link_libraries(mcmc_unittest pthread)
endif()

# Only on Linux: link with openblas and lapack
if (UNIX AND NOT APPLE)
    target_link_libraries(mcmc_unittest
        openblas lapack)
endif()
add_test(mcmc_unittest mcmc_unittest)

######################################################
# Expression Builder Test
######################################################

add_executable(expr_builder_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/expr_builder_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ad_integration_unittest.cpp
    )

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_options(expr_builder_unittest PRIVATE -g -Wall)
else()
	target_compile_options(expr_builder_unittest PRIVATE -g -Wall -Werror -Wextra)
endif()

target_include_directories(expr_builder_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AUTOPPL_INCLUDE_DIRS}
    )
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(expr_builder_unittest gcov)
endif()

target_link_libraries(expr_builder_unittest autoppl_gtest_main ${AUTOPPL_LIBS})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_link_libraries(expr_builder_unittest pthread)
endif()

add_test(expr_builder_unittest expr_builder_unittest)
