#pragma once
#include <utility>
#include <autoppl/util/traits/var_traits.hpp>
#include <autoppl/util/traits/model_expr_traits.hpp>
#include <autoppl/util/packs/offset_pack.hpp>

namespace ppl {
namespace expr {

/**
 * Activates expr with the correct offset values for each parameter
 * and then activates the reference counts for each of the parameters.
 * The total reference count for a parameter should be the number of ParamView
 * objects referencing that parameter in the expression.
 * Every inference algorithm must invoke this call.
 * Otherwise, undefined behavior.
 *
 * @tparam  ExprType    type of expression
 * @param   expr        expression to set parameter offsets
 * @return  offset packs of continuous and discrete parameters, respectively
 */
template <class ExprType>
inline auto activate(ExprType&& expr)
{
    using expr_t = std::decay_t<ExprType>;
    if constexpr (util::is_model_expr_v<expr_t>) {

        util::OffsetPack cont_param_offset;
        util::OffsetPack disc_param_offset;
        auto activate__ = [&](auto& eq_node) {
            auto& var = eq_node.get_variable();
            using var_t = std::decay_t<decltype(var)>;
            if constexpr (util::is_param_v<var_t>) {
                if constexpr (util::var_traits<var_t>::is_cont_v) {
                    var.activate(cont_param_offset);
                } else if constexpr (util::var_traits<var_t>::is_disc_v) {
                    var.activate(disc_param_offset);
                }
            }
        };
        expr.traverse(activate__);
        return std::make_pair(cont_param_offset, disc_param_offset);

    } else if constexpr (util::is_var_expr_v<expr_t>) {

        util::OffsetPack cont_param_offset;
        util::OffsetPack disc_param_offset;
        auto activate__ = [&](auto& eq_node) {
            auto& var = eq_node.get_variable();
            using var_t = std::decay_t<decltype(var)>;
            static_assert(util::is_tparam_v<var_t>);
            if constexpr (util::var_traits<var_t>::is_cont_v) {
                var.activate(cont_param_offset);
            } else if constexpr (util::var_traits<var_t>::is_disc_v) {
                var.activate(disc_param_offset);
            }
        };
        expr.traverse(activate__);

        return std::make_pair(cont_param_offset, disc_param_offset);

    } else {

        static_assert(util::is_model_expr_v<expr_t>,
                      "Expression must be a model or variable expression");
    }
}

} // namespace expr
} // namespace ppl
