#pragma once
#include <random>
#include <autoppl/util/traits/var_traits.hpp>
#include <autoppl/math/math.hpp>

namespace ppl {
namespace expr {

/**
 * Initializes parameters with the given priors and
 * conditional distributions based on the model.
 * Random numbers are generated with gen.
 * Assumes that model was activated and bound before.
 * It only requires binding:
 * - unconstrained
 * - constrained
 * - visit count
 * - transformed parameter values
 */
template <class ProgramType
        , class GenType>
inline void init_params(ProgramType& program, 
                        GenType& gen,
                        bool prune = true,
                        double radius = 2.)
{
    auto& model = program.get_model();

    // default initialization method
    std::uniform_real_distribution cont_dist(-radius, radius);
    auto init_params__ = [&](auto& eq_node) {
        auto& var = eq_node.get_variable();
        using var_t = std::decay_t<decltype(var)>;
        if constexpr (util::is_param_v<var_t>) {
            var.init(gen, cont_dist);
        } 
    };
    model.traverse(init_params__);

    // prune if set to true
    if (!prune) return;

    int n_param_entities = 0;
    auto n_param_entities__ = [&](const auto& eq_node) {
        auto& var = eq_node.get_variable();
        using var_t = std::decay_t<decltype(var)>;
        if constexpr (util::is_param_v<var_t>) {
            ++n_param_entities;
        } 
    };
    model.traverse(n_param_entities__);

    auto log_pdf = program.log_pdf();
    for (int i = 0; i < n_param_entities && log_pdf == math::neg_inf<double>; ++i) {

        // evaluate log-pdf for 2 purposes:
        // 1) evaluates unconstrained to constrained automatically
        // 2) log-pdf needs to be checked that it is not -inf later
        bool modified = false;
        auto prune_params__ = [&](auto& eq_node) {
            auto& var = eq_node.get_variable();
            const auto& dist = eq_node.get_distribution();
            using var_t = std::decay_t<decltype(var)>;
            if constexpr (util::is_param_v<var_t>) {
                bool curr_modified = dist.prune(var, gen);
                if (curr_modified) {
                    var.inv_eval();
                }
                modified = modified || curr_modified;
            } 
        };
        model.traverse(prune_params__);

        // if no unconstrained parameters were modified, log_pdf won't change anymore
        // can early exit 
        if (!modified) break;   

        log_pdf = program.log_pdf();
    }

    assert(log_pdf != math::neg_inf<double>);
}

} // namespace expr
} // namespace ppl
