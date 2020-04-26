#pragma once
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>
#include <array>
#include <variant>
#include <autoppl/util/traits.hpp>
#include <autoppl/variable.hpp>

#define AUTOPPL_MH_UNKNOWN_VALUE_TYPE_ERROR \
    "Unknown value type: must be convertible to util::disc_param_t " \
    "such as uint64_t or util::cont_param_t such as double."

/*
 * Assumptions:
 * - every variable referenced in model is of type Variable<double>
 */

namespace ppl {
namespace details {

/*
 * Convert ValueType to either util::cont_param_t if floating point
 * or util::disc_param_t if integral type.
 * If not either, raises compile-time error.
 */
template <class ValueType, class = void>
struct value_to_param
{
    static_assert(!(std::is_integral_v<ValueType> ||
                    std::is_floating_point_v<ValueType>),
                    AUTOPPL_MH_UNKNOWN_VALUE_TYPE_ERROR);
};
template <class ValueType>
struct value_to_param<ValueType, std::enable_if_t<std::is_integral_v<ValueType>>>
{
    using type = util::disc_param_t;
};
template <class ValueType>
struct value_to_param<ValueType, std::enable_if_t<std::is_floating_point_v<ValueType>>>
{
    using type = util::cont_param_t;
};
template <class ValueType>
using value_to_param_t = typename value_to_param<ValueType>::type;

/*
 * Data structure to keep track of candidates in metropolis-hastings.
 */
struct MHData
{
    std::variant<util::cont_param_t, util::disc_param_t> next;
    // TODO: maybe keep an array for batch sampling?
};

template <class ModelType, class RGenType, class Iter>
inline void mh_posterior__(ModelType& model,
                           Iter params_it,
                           RGenType& gen,
                           size_t n_sample,
                           double curr_log_pdf,
                           double alpha,
                           double stddev)
{
    std::uniform_real_distribution unif_sampler(0., 1.);

    for (size_t iter = 0; iter < n_sample; ++iter) {
        
        size_t n_swaps = 0;                     // during candidate sampling, if sample out-of-bounds,
                                                // traversal will prematurely return and n_swaps < n_params
        bool early_reject = false;              // indicate early sample reject
        double log_alpha = -curr_log_pdf;

        // generate next candidates and place them in parameter
        // variables as next values; update log_alpha
        // The old values are temporary stored in the params vector.
        auto get_candidate = [=, &n_swaps, &early_reject, &gen](auto& eq_node) mutable {
            if (early_reject) return;

            auto& var = eq_node.get_variable();
            using var_t = std::decay_t<decltype(var)>;
            using value_t = typename util::var_traits<var_t>::value_t;
            using state_t = typename util::var_traits<var_t>::state_t;

            if (var.get_state() == state_t::parameter) {
                auto curr = var.get_value(0);
                const auto& dist = eq_node.get_distribution();

                // Choose either continuous or discrete sampler depending on value_t
                if constexpr (std::is_integral_v<value_t>) {
                    std::discrete_distribution disc_sampler({alpha, 1-2*alpha, alpha});
                    auto cand = disc_sampler(gen) - 1 + curr; // new candidate in curr + [-1, 0, 1]
                    // TODO: refactor common logic
                    if (dist.min() <= cand && cand <= dist.max()) { // if within dist bound
                        var.set_value(cand); 
                        ++n_swaps;
                    }
                    else { early_reject = true; return; }
                } else if constexpr (std::is_floating_point_v<value_t>) {
                    std::normal_distribution norm_sampler(static_cast<double>(curr), stddev);
                    auto cand = norm_sampler(gen); 
                    if (dist.min() <= cand && cand <= dist.max()) { // if within dist bound
                        var.set_value(cand); 
                        ++n_swaps;
                    }
                    else { early_reject = true; return; }
                } else {
                    static_assert(!(std::is_integral_v<value_t> ||
                                    std::is_floating_point_v<value_t>), 
                                  AUTOPPL_MH_UNKNOWN_VALUE_TYPE_ERROR);
                }

                // move old value into params
                using converted_value_t = details::value_to_param_t<value_t>;
                params_it->next = static_cast<converted_value_t>(curr);
                ++params_it;
            }
        };
        model.traverse(get_candidate);

        if (early_reject) {

            // swap back original params only up until when candidate was out of bounds.
            auto add_to_storage = [&n_swaps, params_it, iter](auto& eq_node) mutable {
                auto& var = eq_node.get_variable();
                using var_t = std::decay_t<decltype(var)>;
                using value_t = typename util::var_traits<var_t>::value_t;
                using state_t = typename util::var_traits<var_t>::state_t;
                if (var.get_state() == state_t::parameter) {
                    if (n_swaps) {
                        using converted_value_t = details::value_to_param_t<value_t>;
                        var.set_value(std::get<converted_value_t>(params_it->next));
                        ++params_it;
                        --n_swaps;
                    }
                    auto storage = var.get_storage();
                    storage[iter] = var.get_value(0);
                } 
            };
            model.traverse(add_to_storage);
            continue;
        }

        // compute next candidate log pdf and update log_alpha
        double cand_log_pdf = model.log_pdf();
        log_alpha += cand_log_pdf;
        bool accept = (std::log(unif_sampler(gen)) <= log_alpha);

        // If accept, "current" sample for next iteration is already in the variables
        // so simply append to storage.
        // Otherwise, "current" sample for next iteration must be moved back from 
        // params vector into variables.
        auto add_to_storage = [params_it, iter, accept](auto& eq_node) mutable {
            auto& var = eq_node.get_variable();
            using var_t = std::decay_t<decltype(var)>;
            using value_t = typename util::var_traits<var_t>::value_t;
            using state_t = typename util::var_traits<var_t>::state_t;
            if (var.get_state() == state_t::parameter) {
                if (!accept) {
                    using converted_value_t = details::value_to_param_t<value_t>;
                    var.set_value(std::get<converted_value_t>(params_it->next));
                    ++params_it;
                }
                auto storage = var.get_storage();
                storage[iter] = var.get_value(0);
            } 
        };
        model.traverse(add_to_storage);

        // update current log pdf for next iteration
        if (accept) curr_log_pdf = cand_log_pdf;
    }
}

} // namespace details

/*
 * Metropolis-Hastings algorithm to sample from posterior distribution.
 * The posterior distribution is a constant multiple of model.pdf().
 * Any variables that model references which are in state "parameter"
 * is sampled and in state "data" are not.
 * So, model.pdf() is proportional to p(parameters... | data...).
 *
 * User must ensure that they allocated at least as large as n_sample
 * in the storage associated with every parameter referenced in model.
 */
template <class ModelType>
inline void mh_posterior(ModelType& model,
                         double n_sample,
                         double stddev = 1.0,
                         double alpha = 0.25,
                         double seed = std::chrono::duration_cast<
                                        std::chrono::milliseconds>(
                                            std::chrono::system_clock::now().time_since_epoch()
                                            ).count()
                        )
{
    using data_t = details::MHData;
    
    // set-up auxiliary tools
    std::mt19937 gen(seed);
    size_t n_params = 0;
    double curr_log_pdf = 0.;  // current log pdf
    const double initial_radius = 5.;    // arbitrarily chosen radius for initial sampling

    // 1. initialize parameters with values in valid range
    // - discrete valued params sampled uniformly within the distribution range
    // - continuous valued params sampled uniformly within the intersection range
    //   of distribution min and max and [-initial_radius, initial_radius]
    // 2. update n_params with number of parameters
    // 3. compute current log-pdf
    auto init_params = [&](auto& eq_node) {
        auto& var = eq_node.get_variable();
        const auto& dist = eq_node.get_distribution();

        using var_t = std::decay_t<decltype(var)>;
        using value_t = typename util::var_traits<var_t>::value_t;
        using state_t = typename util::var_traits<var_t>::state_t;

        if (var.get_state() == state_t::parameter) {
            if constexpr (std::is_integral_v<value_t>) {
                std::uniform_int_distribution init_sampler(dist.min(), dist.max());
                var.set_value(init_sampler(gen));
            } else if constexpr (std::is_floating_point_v<value_t>) {
                std::uniform_real_distribution init_sampler(
                        std::max(dist.min(), -initial_radius), 
                        std::min(dist.max(), initial_radius)
                        );
                var.set_value(init_sampler(gen));
            } else {
                static_assert(!(std::is_integral_v<value_t> ||
                                std::is_floating_point_v<value_t>), 
                              AUTOPPL_MH_UNKNOWN_VALUE_TYPE_ERROR);
            }
            ++n_params;
        }
        curr_log_pdf += util::log_pdf(var, dist);
    };
    model.traverse(init_params);

    std::vector<data_t> params(n_params);   // vector of parameter-related data with candidate
    details::mh_posterior__(model,
                            params.begin(),
                            gen,
                            n_sample,
                            curr_log_pdf,
                            alpha,
                            stddev);
}

} // namespace ppl
