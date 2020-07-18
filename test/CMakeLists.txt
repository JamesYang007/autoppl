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
    ${CMAKE_CURRENT_SOURCE_DIR}/util/traits/concept_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/traits/dist_expr_traits_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/traits/var_expr_traits_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/traits/var_traits_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/traits/shape_traits_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/iterator/counting_iterator_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/iterator/range_unittest.cpp
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
# Expression Test
######################################################

add_executable(expr_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/variable/param_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/variable/data_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/variable/constant_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/variable/binop_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/variable/dot_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/distribution/bernoulli_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/distribution/normal_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/distribution/uniform_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/model/model_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/expr_builder_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/integration/dist_inttest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/integration/model_inttest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expression/integration/ad_inttest.cpp
    )

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_options(expr_unittest PRIVATE -g -Wall)
else()
	target_compile_options(expr_unittest PRIVATE -g -Wall -Werror -Wextra)
endif()

target_include_directories(expr_unittest PRIVATE
    ${GTEST_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AUTOPPL_INCLUDE_DIRS}
    )
if (AUTOPPL_ENABLE_TEST_COVERAGE)
    target_link_libraries(expr_unittest gcov)
endif()

target_link_libraries(expr_unittest autoppl_gtest_main ${AUTOPPL_LIBS})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_link_libraries(expr_unittest pthread)
endif()

add_test(expr_unittest expr_unittest)

######################################################
# Math Test
######################################################

add_executable(math_unittest
    ${CMAKE_CURRENT_SOURCE_DIR}/math/welford_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/math/density_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/math/math_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/math/autocorrelation_unittest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/math/ess_unittest.cpp
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
