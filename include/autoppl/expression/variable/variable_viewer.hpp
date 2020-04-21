#pragma once
#include <autoppl/util/var_traits.hpp>
#include <autoppl/util/var_expr_traits.hpp>

namespace ppl {
namespace expr {

/*
 * VariableViewer is a viewer of some variable type. 
 * It will mainly be used to view Variable class defined in autoppl/variable.hpp.
 */
template <class VariableType>
struct VariableViewer
{
    static_assert(util::is_var_v<VariableType>);

    using var_t = VariableType;
    using value_t = typename util::var_traits<var_t>::value_t;

    VariableViewer(var_t& var)
        : var_ref_{var}
    {}

    operator value_t() const { return get_value(); }

    value_t get_value() const 
    { return static_cast<value_t>(var_ref_.get()); }

private:
    using var_ref_t = std::reference_wrapper<var_t>;
    var_ref_t var_ref_;
};

} // namespace expr
} // namespace ppl
