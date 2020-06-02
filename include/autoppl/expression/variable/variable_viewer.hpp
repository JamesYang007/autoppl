#pragma once
#include <algorithm>
#include <fastad>
#include <autoppl/util/var_traits.hpp>
#include <autoppl/util/var_expr_traits.hpp>

namespace ppl {
namespace expr {

/**
 * VariableViewer is a viewer of some variable type. 
 * It will mainly be used to view Variable class defined in autoppl/variable.hpp.
 */
template <class VariableType>
struct VariableViewer : util::VarExpr<VariableViewer<VariableType>>
{
    static_assert(util::assert_is_var_v<VariableType>);

    using var_t = VariableType;
    using value_t = typename util::var_traits<var_t>::value_t;

    VariableViewer(var_t& var)
        : var_ref_{var}
    {}

    value_t get_value(size_t i = 0) const { return var_ref_.get().get_value(i); }
    size_t size() const { return var_ref_.get().size(); }

    /**
     * Returns ad expression of the variable.
     * If variable is parameter, find from vars and return.
     * Otherwise if data, return idx'th ad::constant of that value.
     */
    template <class VecRefType, class VecADVarType>
    auto get_ad(const VecRefType& keys,
                const VecADVarType& vars,
                size_t idx = 0) const
    {
        if constexpr (util::is_param_v<var_t>) {
            static_cast<void>(idx);
            const void* addr = &var_ref_.get();
            auto it = std::find(keys.begin(), keys.end(), addr);
            assert(it != keys.end());
            size_t i = std::distance(keys.begin(), it);
            return vars[i];
        } else if constexpr (util::is_data_v<var_t>) {
            return ad::constant(this->get_value(idx));
        }
    }

private:
    using var_ref_t = std::reference_wrapper<var_t>;
    var_ref_t var_ref_;
};

} // namespace expr
} // namespace ppl
