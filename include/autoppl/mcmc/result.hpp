#pragma once
#include <string>
#include <Eigen/Dense>
#include <autoppl/util/traits/dist_expr_traits.hpp>

namespace ppl {

/**
 * MCMC Output object when a sampling algorithm returns.
 * Stores useful information from MCMC sampling routine such as:
 * - continuous and discrete samples
 * - warmup and sampling time
 * - name of mcmc algorithm invoked
 */
template <int Major = Eigen::ColMajor>
struct MCMCResult
{
    using cont_samples_t = Eigen::Matrix<util::cont_param_t, Eigen::Dynamic, Eigen::Dynamic, Major>;
    using disc_samples_t = Eigen::Matrix<util::disc_param_t, Eigen::Dynamic, Eigen::Dynamic>;

    cont_samples_t cont_samples;
    disc_samples_t disc_samples;
    std::string name;
    double warmup_time = 0;
    double sampling_time = 0;

    MCMCResult() =default;
    MCMCResult(size_t n_samples,
               size_t n_cont_params,
               size_t n_disc_params)
        : cont_samples(n_samples, n_cont_params)
        , disc_samples(n_samples, n_disc_params)
    {}
};

} // namespace ppl
