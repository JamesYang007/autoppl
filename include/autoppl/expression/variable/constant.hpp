#pragma once
#include <autoppl/util/var_expr_traits.hpp>

namespace ppl {
namespace expr {

template <class ValueType>
struct Constant : util::VarExpr<Constant<ValueType>>
{
    using value_t = ValueType;
    Constant(value_t c)
        : c_{c}
    {}
    explicit operator value_t() const { return get_value(); }
    value_t get_value() const { return c_; }

private:
    value_t c_;
};

} // namespace expr
} // namespace ppl
