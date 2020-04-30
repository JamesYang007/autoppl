######################################################
# Examples
######################################################

# Model Size
add_executable(model_size
    model_size.cpp
    )
target_compile_options(model_size PRIVATE -g -Wall -Werror -Wextra)
target_link_libraries(model_size ${PROJECT_NAME} FastAD::FastAD)

# Sample from standard normal
add_executable(sample_std_normal
    sample_std_normal.cpp
    )
target_compile_options(sample_std_normal PRIVATE -g -Wall -Werror -Wextra)
target_link_libraries(sample_std_normal ${PROJECT_NAME} FastAD::FastAD)

# Sample from some joint distribution
add_executable(sample_joint_dist
    sample_joint_dist.cpp
    )
target_compile_options(sample_joint_dist PRIVATE -g -Wall -Werror -Wextra)
target_link_libraries(sample_joint_dist ${PROJECT_NAME} FastAD::FastAD)

# Normal data with posterior on mean and stddev
add_executable(normal_posterior_mean_stddev
    normal_posterior_mean_stddev.cpp
    )
target_compile_options(normal_posterior_mean_stddev PRIVATE
    -g -Wall -Werror -Wextra -DARMA_DONT_USE_WRAPPER)
target_link_libraries(normal_posterior_mean_stddev
    ${PROJECT_NAME} FastAD::FastAD ${ARMADILLO_LIBRARIES})
