#pragma once
#include <cstddef>
#include <autoppl/mcmc/hmc/nuts/step_adapter.hpp>
#include <autoppl/mcmc/hmc/momentum_handler.hpp>
#include <autoppl/mcmc/sampler_tools.hpp>

namespace ppl {

/*
 * User configuration for NUTS algorithm.
 */
template <class VarAdapterPolicy=diag_var>
struct NUTSConfig
{
    using var_adapter_policy_t = VarAdapterPolicy;

    // configuration for sampling
    size_t warmup = 1000;
    size_t n_samples = 1000;
    size_t seed = mcmc::random_seed();
    size_t max_depth = 10;

    // configuration for step-size adaptation
    StepConfig step_config;

    // configuration for variance adaptation
    VarConfig var_config;
};

/*
 * Traits to get member aliases of any NUTS config object.
 */
template <class NUTSConfigType>
struct nuts_config_traits
{
    using var_adapter_policy_t = 
        typename NUTSConfigType::var_adapter_policy_t;
};

} // namespace ppl
