#pragma once
#include <random>
#include <autoppl/util/var_traits.hpp>

#define AUTOPPL_MH_UNKNOWN_VALUE_TYPE_ERROR \
    "Unknown value type: must be convertible to util::disc_param_t " \
    "such as uint64_t or util::cont_param_t such as double."

namespace ppl {
namespace alg {

/*
 * Returns number of parameters in the model.
 * Note that this assumes every parameter is univariate.
 */
template <class ModelType>
size_t get_n_params(const ModelType& model)
{
    size_t n = 0;
    auto get_n_params__ = [&](const auto& eq_node) {
        const auto& var = eq_node.get_variable();
        using var_t = std::decay_t<decltype(var)>;
        using state_t = typename util::var_traits<var_t>::state_t;
        n += (var.get_state() == state_t::parameter);
    };
    model.traverse(get_n_params__);
    return n;
}

/*
 * Initializes parameters with the given priors and
 * conditional distributions based on the model.
 * Random numbers are generated with gen.
 */
template <class ModelType, class GenType>
void init_params(ModelType& model, GenType& gen)
{
    // arbitrarily chosen radius for initial sampling
    constexpr double initial_radius = 5.;    

    auto init_params__ = [&](auto& eq_node) {
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
        }
    };
    model.traverse(init_params__);
}

} // namespace alg
} // namespace ppl
