######################################################
# Examples
######################################################

# Model Size
add_executable(model_size
    model_size.cpp
    )
target_compile_options(model_size PRIVATE -g -Wall -Werror -Wextra)
target_include_directories(model_size
    PRIVATE ${AUTOPPL_INCLUDE_DIRS})
target_link_libraries(model_size ${AUTOPPL_LIBS})

# Sample from standard normal
add_executable(sample_std_normal
    sample_std_normal.cpp
    )
target_compile_options(sample_std_normal PRIVATE -g -Wall -Werror -Wextra)
target_include_directories(sample_std_normal
    PRIVATE ${AUTOPPL_INCLUDE_DIRS})
target_link_libraries(sample_std_normal ${AUTOPPL_LIBS})

# Sample from some joint distribution
add_executable(sample_joint_dist
    sample_joint_dist.cpp
    )
target_compile_options(sample_joint_dist PRIVATE -g -Wall -Werror -Wextra)
target_include_directories(sample_joint_dist
    PRIVATE ${AUTOPPL_INCLUDE_DIRS})
target_link_libraries(sample_joint_dist ${AUTOPPL_LIBS})

# Normal data with posterior on mean and stddev
add_executable(normal_posterior_mean_stddev
    normal_posterior_mean_stddev.cpp
    )
target_compile_options(normal_posterior_mean_stddev PRIVATE
    -g -Wall -Werror -Wextra)
target_include_directories(normal_posterior_mean_stddev
    PRIVATE ${AUTOPPL_INCLUDE_DIRS})
target_link_libraries(normal_posterior_mean_stddev ${AUTOPPL_LIBS})
