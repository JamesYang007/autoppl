#pragma once
#include <type_traits>
#include <functional>
#include <autoppl/util/var_traits.hpp>
#include <autoppl/util/model_expr_traits.hpp>
#include <autoppl/util/dist_expr_traits.hpp>

namespace ppl {
namespace expr {

/*
 * This class represents a "node" in the model expression
 * that relates a var with a distribution.
 */
template <class VarType, class DistType>
struct EqNode : util::ModelExpr<EqNode<VarType, DistType>>
{
    static_assert(util::assert_is_var_v<VarType>);
    static_assert(util::assert_is_dist_expr_v<DistType>);

    using var_t = VarType;
    using dist_t = DistType;
    using dist_value_t = typename util::dist_expr_traits<dist_t>::dist_value_t;

    EqNode(var_t& var, 
           const dist_t& dist) noexcept
        : orig_var_ref_{var}
        , dist_{dist}
    {}

    /* 
     * Generic traversal function.
     * Assumes that eq_f is a function that takes in 1 parameter,
     * which is simply the current object.
     */
    template <class EqNodeFunc>
    void traverse(EqNodeFunc&& eq_f)
    {
        using this_t = EqNode<VarType, DistType>;
        eq_f(static_cast<this_t&>(*this));
    }

    /*
     * Compute pdf of underlying distribution with underlying value.
     * Assumes that underlying value has been assigned properly.
     */
    dist_value_t pdf() const
    { return dist_.pdf(orig_var_ref_.get().get_value()); }

    /*
     * Compute log-pdf of underlying distribution with underlying value.
     * Assumes that underlying value has been assigned properly.
     */
    dist_value_t log_pdf() const
    { return dist_.log_pdf(orig_var_ref_.get().get_value()); }

    auto& get_variable() { return orig_var_ref_.get(); }
    const auto& get_distribution() const { return dist_; }

private:
    using var_ref_t = std::reference_wrapper<var_t>;    
    var_ref_t orig_var_ref_;      // reference of the original var since 
                                  // any configuration may be changed until right before update 
    dist_t dist_;                 // distribution associated with var
};

} // namespace expr
} // namespace ppl
