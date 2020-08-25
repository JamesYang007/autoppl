add_executable(mh_bayes_net ${CMAKE_CURRENT_SOURCE_DIR}/mh_bayes_net.cpp)
target_include_directories(mh_bayes_net PRIVATE ${GBENCH_DIR}/include ${AUTOPPL_INCLUDE_DIRS})
target_link_libraries(mh_bayes_net benchmark benchmark_main ${AUTOPPL_LIBS})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_link_libraries(mh_bayes_net pthread)
endif()

add_executable(normal_wishart_prior ${CMAKE_CURRENT_SOURCE_DIR}/normal_wishart_prior.cpp)
target_include_directories(normal_wishart_prior PRIVATE ${GBENCH_DIR}/include ${AUTOPPL_INCLUDE_DIRS})
target_link_libraries(normal_wishart_prior benchmark benchmark_main ${AUTOPPL_LIBS})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_link_libraries(normal_wishart_prior pthread)
endif()

add_executable(normal_two_prior_distribution ${CMAKE_CURRENT_SOURCE_DIR}/normal_two_prior_distribution.cpp)
target_include_directories(normal_two_prior_distribution PRIVATE ${GBENCH_DIR}/include ${AUTOPPL_INCLUDE_DIRS})
target_link_libraries(normal_two_prior_distribution benchmark benchmark_main ${AUTOPPL_LIBS})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_link_libraries(normal_two_prior_distribution pthread)
endif()

add_executable(regression_autoppl ${CMAKE_CURRENT_SOURCE_DIR}/regression_autoppl.cpp)
target_include_directories(regression_autoppl PRIVATE ${GBENCH_DIR}/include ${AUTOPPL_INCLUDE_DIRS})
target_link_libraries(regression_autoppl benchmark benchmark_main ${AUTOPPL_LIBS})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_link_libraries(regression_autoppl pthread)
endif()
add_custom_command(
        TARGET regression_autoppl POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/life-clean.csv
                ${CMAKE_CURRENT_BINARY_DIR}/life-clean.csv)

add_executable(regression_autoppl_2 ${CMAKE_CURRENT_SOURCE_DIR}/regression_autoppl_2.cpp)
target_include_directories(regression_autoppl_2 PRIVATE ${GBENCH_DIR}/include ${AUTOPPL_INCLUDE_DIRS})
target_link_libraries(regression_autoppl_2 benchmark benchmark_main ${AUTOPPL_LIBS})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_link_libraries(regression_autoppl_2 pthread)
endif()

add_executable(stochastic_volatility ${CMAKE_CURRENT_SOURCE_DIR}/stochastic_volatility.cpp)
target_include_directories(stochastic_volatility PRIVATE ${GBENCH_DIR}/include ${AUTOPPL_INCLUDE_DIRS})
target_link_libraries(stochastic_volatility benchmark benchmark_main ${AUTOPPL_LIBS})
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_link_libraries(stochastic_volatility pthread)
endif()
