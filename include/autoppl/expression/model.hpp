#pragma once
#include <type_traits>
#include <functional>
#include <optional>
#include <autoppl/util/traits.hpp>
#include <autoppl/expression/model_expr.hpp>

namespace ppl {
namespace expr {
namespace details {

template <class Iter>
struct IdentityVarFunctor
{
    using value_t = typename std::iterator_traits<Iter>::value_type;
    value_t& operator()(value_t& var)
    { return var; }
};

} // namespace details

/*
 * This class represents a "node" in the model expression
 * that relates a var with a distribution.
 */
template <class VarType, class DistType>
struct EqNode : public ModelExpr<EqNode<VarType, DistType>>
{
    using var_t = VarType;
    using dist_t = DistType;
    using dist_value_t = typename dist_traits<dist_t>::dist_value_t;

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

/*
 * This class represents a "node" in a model expression that
 * "glues" two sub-model expressions.
 */
template <class LHSNodeType, class RHSNodeType>
struct GlueNode : public ModelExpr<GlueNode<LHSNodeType, RHSNodeType>>
{
    using left_node_t = LHSNodeType;
    using right_node_t = RHSNodeType;
    using dist_value_t = std::common_type_t<
        typename node_traits<left_node_t>::dist_value_t,
        typename node_traits<right_node_t>::dist_value_t
            >;

    GlueNode(const left_node_t& lhs,
             const right_node_t& rhs) noexcept
        : left_node_{lhs}
        , right_node_{rhs}
    {}

    /* 
     * Generic traversal function.
     * Recursively traverses left then right.
     */
    template <class EqNodeFunc>
    void traverse(EqNodeFunc&& eq_f)
    {
        left_node_.traverse(eq_f);
        right_node_.traverse(eq_f);
    }

    /*
     * Computes left node joint pdf then right node joint pdf
     * and returns the product of the two.
     */
    dist_value_t pdf() const
    { return left_node_.pdf() * right_node_.pdf(); }

    /*
     * Computes left node joint log-pdf then right node joint log-pdf
     * and returns the sum of the two.
     */
    dist_value_t log_pdf() const
    { return left_node_.log_pdf() + right_node_.log_pdf(); }

private:
    left_node_t left_node_;
    right_node_t right_node_;
};

} // namespace expr
} // namespace ppl
