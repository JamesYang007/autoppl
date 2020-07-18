#pragma once
#include <cstddef>
#include <type_traits>
#include <autoppl/util/traits/var_traits.hpp>

namespace ppl {
namespace expr {

/**
 * Activates cache offsets of model for any distribution
 * or variable expression which require caching.
 * Any inference algorithm intending to use AD must invoke this call
 * before proceeding.
 * 
 * @tparam  ModelType   type of model expression
 * @param   model       model expression to set cache offsets
 * @return  size of cache required by model
 */
template <class ModelType>
inline size_t activate_cache(ModelType&& model)
{
    size_t cache_offset = 0;
    auto activate__ = [&](auto& eq_node) {
        auto& dist = eq_node.get_distribution();
        cache_offset = dist.set_cache_offset(cache_offset);
    };
    model.traverse(activate__);
    return cache_offset;
}

/**
 * Activates model with the correct offset values for each parameter
 * and cache offset (if needed) by any distribution or variable expressions.
 * Every inference algorithm must invoke this call.
 * Otherwise, undefined behavior.
 *
 * @tparam  ModelType   type of model expression
 * @param   model       model expression to set parameter offsets
 * @return  size of parameters
 */
template <class ModelType>
inline size_t activate(ModelType&& model)
{
    size_t param_offset = 0;
    auto activate__ = [&](auto& eq_node) {
        auto& var = eq_node.get_variable();
        using var_t = std::decay_t<decltype(var)>;
        if constexpr (util::is_param_v<var_t>) {
            param_offset = var.set_offset(param_offset);
        }
    };
    model.traverse(activate__);
    return param_offset;
}

} // namespace expr
} // namespace ppl
