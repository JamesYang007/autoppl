#pragma once
#include <chrono>
#include <random>
#include <fastad>
#include <armadillo>
#include <autoppl/expression/model/model_utils.hpp>
#include <autoppl/util/var_traits.hpp>

#define AUTOPPL_MH_UNKNOWN_VALUE_TYPE_ERROR \
    "Unknown value type: must be convertible to util::disc_param_t " \
    "such as uint64_t or util::cont_param_t such as double."

namespace ppl {
namespace mcmc {

/**
 * Get current time in milliseconds for random seeding.
 */
inline size_t random_seed()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::system_clock::now().time_since_epoch()).count();
}

/**
 * Initializes parameters with the given priors and
 * conditional distributions based on the model.
 * Random numbers are generated with gen.
 */
template <class ModelType, class GenType>
void init_params(ModelType& model, GenType& gen)
{
    // arbitrarily chosen radius for initial sampling
    constexpr double initial_radius = 2.;    

    auto init_params__ = [&](auto& eq_node) {
        auto& var = eq_node.get_variable();
        const auto& dist = eq_node.get_distribution();

        using var_t = std::decay_t<decltype(var)>;
        using value_t = typename util::var_traits<var_t>::value_t;

#if __cplusplus <= 201703L
        if constexpr (util::is_param_v<var_t>) {
#else
        if constexpr (util::param<var_t>) {
#endif

            if constexpr (std::is_integral_v<value_t>) {
                std::uniform_int_distribution init_sampler(dist.min(), dist.max());
                var.set_value(init_sampler(gen));

            } else if constexpr (std::is_floating_point_v<value_t>) {
                std::uniform_real_distribution init_sampler(-initial_radius, initial_radius);

                // if unbounded prior
                if (dist.min() == std::numeric_limits<value_t>::lowest() &&
                    dist.max() == std::numeric_limits<value_t>::max()) {
                    var.set_value(init_sampler(gen));
                }

                // TODO: uncomment once there exists distributions with these properties
                //// if bounded above but not below
                //else if (dist.min() == std::numeric_limits<value_t>::lowest()) {
                //    var.set_value(dist.max() - std::exp(init_sampler(gen)));
                //}

                //// if bounded below but not above
                //else if (dist.max() == std::numeric_limits<value_t>::max()) {
                //    var.set_value(std::exp(init_sampler(gen)) + dist.min());
                //}
                
                // bounded below and above
                else {
                    value_t range = dist.max() - dist.min();
                    value_t avg = dist.min() + range / 2.;
                    var.set_value(avg + range / (2 * initial_radius) * init_sampler(gen));
                }

            } else {
                static_assert(!(std::is_integral_v<value_t> ||
                                std::is_floating_point_v<value_t>), 
                              AUTOPPL_MH_UNKNOWN_VALUE_TYPE_ERROR);
            }
        }
    };
    model.traverse(init_params__);
}

/**
 * Initializes first sample of parameters using mcmc::init_params.
 * Helper function to copy the samples into theta_curr.
 */
template <class ModelType
        , class MatType
        , class GenType>
void init_sample(ModelType& model,
                 MatType& theta_curr,
                 GenType& gen)
{
    mcmc::init_params(model, gen);    
    auto theta_curr_it = theta_curr.begin();
    auto copy_params_potential = [&](const auto& eq_node) {
        const auto& var = eq_node.get_variable();
        using var_t = std::decay_t<decltype(var)>;
#if __cplusplus <= 201703L
        if constexpr (util::is_param_v<var_t>) {
#else
        if constexpr (util::param<var_t>) {
#endif
            *theta_curr_it = var.get_value(); 
            ++theta_curr_it;
        }
    };
    model.traverse(copy_params_potential);
}

/**
 * Get unique raw addresses of the referenced variables in the model.
 * Can be used to bind algorithm specific storage associated with each variable.
 */
template <class ModelType>
void get_keys(const ModelType& model,
              std::vector<const void*>& keys)
{
    constexpr size_t n_params = get_n_params_v<ModelType>;
    keys.resize(n_params);
    auto keys_it = keys.begin();
    auto get_keys = [&](auto& eq_node) {
        auto& var = eq_node.get_variable();
        using var_t = std::decay_t<decltype(var)>;
#if __cplusplus <= 201703L
        if constexpr (util::is_param_v<var_t>) {
#else
        if constexpr (util::param<var_t>) {
#endif
            *keys_it = &var;
            ++keys_it;
        }
    };
    model.traverse(get_keys);
}

/**
 * Store ith sample currently in theta_curr into 
 * storage by traversing model.
 */
template <class ModelType, class MatType>
void store_sample(ModelType& model,
                  MatType& theta_curr,
                  size_t i)
{
    auto theta_curr_it = theta_curr.begin();
    auto store_sample = [&, i](auto& eq_node) {
        auto& var = eq_node.get_variable();
        using var_t = std::decay_t<decltype(var)>;
#if __cplusplus <= 201703L
        if constexpr (util::is_param_v<var_t>) {
#else
        if constexpr (util::param<var_t>) {
#endif
            auto storage_ptr = var.get_storage();
            storage_ptr[i] = *theta_curr_it;
            ++theta_curr_it;
        }
    };
    model.traverse(store_sample);
}

/**
 * Accepts or rejects with given probability using UniformDistType
 * object that works with GenType.
 * The uniform sampler must sample from [0,1].
 */
template <class UniformDistType, class GenType>
bool accept_or_reject(double p, 
                      UniformDistType&& unif_sampler,
                      GenType&& gen)
{
    double u = unif_sampler(gen);
    return (u <= p);
}

} // namespace mcmc
} // namespace ppl
