#pragma once
#include <algorithm>
#include <array>
#include <autoppl/util/logging.hpp>
#include <autoppl/util/traits/traits.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <variant>
#include <vector>
#include <autoppl/mcmc/sampler_tools.hpp>
#include <autoppl/expression/activate.hpp>

/**
 * Assumptions:
 * - every variable referenced in model is of type Variable<double>
 */

namespace ppl {
namespace mcmc {
namespace details {

/**
 * Convert ValueType to either util::cont_param_t if floating point
 * or util::disc_param_t if integral type.
 * If not either, raises compile-time error.
 */
template <class ValueType, class = void>
struct value_to_param
{
    static_assert(!(util::is_cont_v<ValueType> ||
                    util::is_disc_v<ValueType>),
                    PPL_CONT_XOR_DISC);
};
template <class ValueType>
struct value_to_param<ValueType, 
    std::enable_if_t<util::is_disc_v<ValueType>> >
{
    using type = util::disc_param_t;
};
template <class ValueType>
struct value_to_param<ValueType, 
    std::enable_if_t<util::is_cont_v<ValueType>> >
{
    using type = util::cont_param_t;
};
template <class ValueType>
using value_to_param_t = typename value_to_param<ValueType>::type;

/**
 * Data structure to keep track of candidates in metropolis-hastings.
 */
struct MHData
{
    std::variant<util::cont_param_t, util::disc_param_t> curr;
    std::variant<util::cont_param_t, util::disc_param_t> next;
};

// Helper functor to get the correct variant value.
struct get_curr
{
    template <class ValueType, class MHDataType>
    constexpr auto&& operator()(MHDataType&& d) noexcept
    { return *std::get_if<ValueType>(&d.curr); }
};

} // namespace details

template <class ModelType
        , class PVecType
        , class RGenType>
inline void mh__(const ModelType& model,
                 PVecType& pvalues,
                 RGenType& gen,
                 size_t n_sample,
                 size_t warmup,
                 double curr_log_pdf,
                 double alpha,
                 double stddev)
{
    std::uniform_real_distribution metrop_sampler(0., 1.);
    std::discrete_distribution disc_sampler({alpha, 1-2*alpha, alpha});
    std::normal_distribution norm_sampler(0., stddev);

    auto logger = util::ProgressLogger(n_sample + warmup, "Metropolis-Hastings");

    for (size_t iter = 0; iter < n_sample + warmup; ++iter) {
        logger.printProgress(iter);

        size_t n_swaps = 0;                     // during candidate sampling, if sample out-of-bounds,
                                                // traversal will prematurely return and n_swaps < n_params
        bool early_reject = false;              // indicate early sample reject
        double log_alpha = -curr_log_pdf;

        // generate next candidates and place them in parameter
        // variables as next values; update log_alpha
        auto get_candidate = [&](const auto& eq_node) mutable {
            if (early_reject) return;

            const auto& var = eq_node.get_variable();
            const auto& dist = eq_node.get_distribution();
            using var_t = std::decay_t<decltype(var)>;
            using value_t = typename util::var_traits<var_t>::value_t;
            using converted_value_t = details::value_to_param_t<value_t>;

            if constexpr (util::is_param_v<var_t>) {
                // generate next candidates for each element of parameter
                for (size_t i = 0; i < var.size(); ++i) {
                    auto& pstate = var.value(pvalues, i);   // MHData object corresponding to ith param elt
                    converted_value_t& curr_val = *std::get_if<converted_value_t>(&pstate.curr);
                    converted_value_t& next_val = *std::get_if<converted_value_t>(&pstate.next);

                    converted_value_t min = dist.min(pvalues, i, details::get_curr());
                    converted_value_t max = dist.max(pvalues, i, details::get_curr());

                    // choose delta based on if discrete or continuous param
                    if constexpr (util::is_disc_v<var_t>) 
                    { next_val = curr_val + disc_sampler(gen) - 1; } 
                    else { next_val = curr_val + norm_sampler(gen); } 

                    if (min <= next_val && next_val <= max) { // if within dist bound
                        std::swap(pstate.curr, pstate.next);
                        ++n_swaps;
                    } else { early_reject = true; return; }

                } // end for
            }
        };
        model.traverse(get_candidate);

        if (early_reject) {

            // swap back original params only up until when candidate was out of bounds.
            for (size_t i = 0; i < n_swaps; ++i) {
                std::swap(pvalues[i].curr, pvalues[i].next);
            }

        } else {

            // compute next candidate log pdf and update log_alpha
            double cand_log_pdf = model.log_pdf(pvalues, details::get_curr());
            log_alpha += cand_log_pdf;
            bool accept = (std::log(metrop_sampler(gen)) <= log_alpha);

            // if not accept, "current" sample for next iteration is in next: swap the two!
            if (!accept) {
                for (auto& pvalue : pvalues) {
                    std::swap(pvalue.curr, pvalue.next);
                }
            } else {
                // update current log pdf for next iteration
                curr_log_pdf = cand_log_pdf;
            }

        }

        if (iter >= warmup) {
            store_sample(model, pvalues, 
                         iter-warmup, details::get_curr());
        }
    }

    std::cout << std::endl;
}

} // namespace mcmc

/**
 * Metropolis-Hastings algorithm to sample from posterior distribution.
 * The posterior distribution is a constant multiple of model.pdf().
 * Any variables that model references which are Params
 * are sampled but Data variables are ignored.
 * So, model.pdf() is proportional to p(parameters... | data...).
 *
 * User must ensure that they allocated at least as large as n_sample
 * in the storage associated with every parameter referenced in model.
 */
template <class ModelType>
inline void mh(ModelType& model,
               double n_sample,
               size_t warmup = 1000,
               double stddev = 1.0,
               double alpha = 0.25,
               size_t seed = mcmc::random_seed())
{
    using data_t = mcmc::details::MHData;
    
    // REALLY important
    size_t n_params = expr::activate(model);

    // data structure to keep track of param candidates
    std::vector<data_t> params(n_params);   // vector of parameter-related data with candidate

    // initialize sample 0
    std::mt19937 gen(seed);
    mcmc::init_params(model, gen, params, mcmc::details::get_curr());

    // compute log pdf with sample 0
    double curr_log_pdf = model.log_pdf(params, mcmc::details::get_curr());

    // sample the rest
    mcmc::mh__(model,
               params,
               gen,
               n_sample,
               warmup,
               curr_log_pdf,
               alpha,
               stddev);
}

} // namespace ppl
