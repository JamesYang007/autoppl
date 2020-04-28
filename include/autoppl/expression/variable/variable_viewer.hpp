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
struct VariableViewer : util::VarExpr<VariableViewer<VariableType>>
{
    static_assert(util::assert_is_var_v<VariableType>);

    using var_t = VariableType;
    using value_t = typename util::var_traits<var_t>::value_t;

    VariableViewer(var_t& var)
        : var_ref_{var}
    {}

    value_t get_value(size_t i) const { return var_ref_.get().get_value(i); }
    size_t size() const { return var_ref_.get().size(); }

private:
    using var_ref_t = std::reference_wrapper<var_t>;
    var_ref_t var_ref_;
};

} // namespace expr
} // namespace ppl
