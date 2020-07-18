#pragma once
#include <fastad_bits/node.hpp>
#include <autoppl/util/traits/var_expr_traits.hpp>
#include <autoppl/util/functional.hpp>

#define PPL_CONSTANT_SHAPE_UNSUPPORTED \
    "Unsupported shape for constants. "

namespace ppl {
namespace expr {

template <class ValueType
        , class ShapeType=ppl::scl>
struct Constant
{
    static_assert(util::is_scl_v<ShapeType>,
                  PPL_CONSTANT_SHAPE_UNSUPPORTED);
};

template <class ValueType>
struct Constant<ValueType, ppl::scl>:
    util::VarExprBase<Constant<ValueType, ppl::scl>>
{
    using value_t = ValueType;
    using shape_t = ppl::scl;
    using index_t = uint32_t;
    static constexpr bool has_param = false;
    static constexpr size_t fixed_size = 1;

    Constant(value_t c) : c_{c} {}

    template <class PVecType
            , class F = util::identity>
    value_t value(const PVecType&, 
                  size_t=0,
                  F = F()) const 
    { return c_; }

    constexpr size_t size() const { return fixed_size; }

    template <class VecADVarType>
    auto to_ad(const VecADVarType&,
               const VecADVarType&,
               size_t = 0) const
    { return ad::constant(c_); }

    index_t set_cache_offset(index_t idx) const 
    { return idx; }

private:
    value_t c_;
};

} // namespace expr
} // namespace ppl

#undef PPL_CONSTANT_VEC_UNSUPPORTED 
#undef PPL_CONSTANT_MAT_UNSUPPORTED 
