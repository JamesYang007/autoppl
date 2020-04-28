#pragma once
#include <autoppl/util/var_expr_traits.hpp>
#include <fastad>

namespace ppl {
namespace expr {

template <class ValueType>
struct Constant : util::VarExpr<Constant<ValueType>>
{
    using value_t = ValueType;
    Constant(value_t c)
        : c_{c}
    {}
    value_t get_value(size_t = 0) const {
        return c_;
    }

    constexpr size_t size() const { return 1; }

    /* 
     * Returns ad expression of the constant.
     */
    template <class VecRefType, class VecADVarType>
    auto get_ad(const VecRefType&,
                const VecADVarType&) const
    { return ad::constant(c_); }

private:
    value_t c_;
};

} // namespace expr
} // namespace ppl
