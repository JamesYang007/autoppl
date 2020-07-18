#pragma once
#include <chrono>
#include <random>
#include <fastad>
#include <armadillo>
#include <autoppl/util/traits/var_traits.hpp>
#include <autoppl/util/iterator/counting_iterator.hpp>
#include <autoppl/util/functional.hpp>
#include <autoppl/math/math.hpp>

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
 * Assumes that model was initialized before.
 */
template <class ModelType
        , class GenType
        , class PVecType
        , class F = util::identity>
inline void init_params(const ModelType& model, 
                        GenType& gen,
                        PVecType& pvalues,
                        F f = F())
{
    // arbitrarily chosen radius for initial sampling
    constexpr double initial_radius = 2.;    

    // initialize each parameter
    auto init_params__ = [&](const auto& eq_node) {
        const auto& var = eq_node.get_variable();
        const auto& dist = eq_node.get_distribution();

        using var_t = std::decay_t<decltype(var)>;
        using value_t = typename util::var_traits<var_t>::value_t;

        if constexpr (util::is_param_v<var_t>) {

            // initialization routine for each element of that parameter
            for (size_t i = 0; i < var.size(); ++i) {

                auto min = dist.min(pvalues, i, f);
                auto max = dist.max(pvalues, i, f);

                if constexpr (util::var_traits<var_t>::is_disc_v) {
                    std::uniform_int_distribution init_sampler(min, max);
                    auto new_val = init_sampler(gen);
                    var.value(pvalues, i, f) = new_val;

                } else {
                    std::uniform_real_distribution init_sampler(-initial_radius, initial_radius);

                    // if unbounded prior
                    if (min == math::neg_inf<value_t> &&
                        max == math::inf<value_t>) {
                        var.value(pvalues, i, f) = init_sampler(gen);
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
                        value_t range = max - min;
                        value_t avg = min + range / 2.;
                        var.value(pvalues, i, f) = 
                            avg + range / (2 * initial_radius) * init_sampler(gen);
                    }

                } // end outer else         
            } // end for
        } // end if 

    };
    model.traverse(init_params__);
}

/**
 * Store ith sample currently in theta_curr into 
 * storage by traversing model.
 * Assumes that theta_curr[i] is the value of the ith parameter in model.
 * If the parameter is a vector and theta_curr[i] is the value for the first
 * element of the parameter, theta_curr[i+j] is the jth value within the parameter.
 */
template <class ModelType
        , class MatType
        , class F = util::identity>
inline void store_sample(const ModelType& model,
                         const MatType& theta_curr,
                         size_t i,
                         F f = F())
{
    auto store_sample__ = [&, i](const auto& eq_node) {
        const auto& var = eq_node.get_variable();
        using var_t = std::decay_t<decltype(var)>;
        if constexpr (util::is_param_v<var_t>) {
            for (size_t j = 0; j < var.size(); ++j) {
                auto var_val = var.value(theta_curr, j, f);
                auto storage_ptr = var.storage(j);
                storage_ptr[i] = var_val;
            }
        }
    };
    model.traverse(store_sample__);
}

/**
 * Accepts or rejects with given probability using UniformDistType
 * object that works with GenType.
 * The uniform sampler must sample from [0,1].
 */
template <class UniformDistType, class GenType>
inline bool accept_or_reject(double p, 
                             UniformDistType&& unif_sampler,
                             GenType&& gen)
{
    return (unif_sampler(gen) <= p);
}

} // namespace mcmc
} // namespace ppl
