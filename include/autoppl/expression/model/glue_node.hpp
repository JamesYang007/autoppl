#pragma once
#include <type_traits>
#include <functional>
#include <autoppl/util/traits/model_expr_traits.hpp>
#include <autoppl/util/functional.hpp>

namespace ppl {
namespace expr {

/**
 * This class represents a "node" in a model expression that
 * "glues" two sub-model expressions.
 */
template <class LHSNodeType
        , class RHSNodeType>
struct GlueNode: util::ModelExprBase<GlueNode<LHSNodeType, RHSNodeType>>
{
    static_assert(util::is_model_expr_v<LHSNodeType>);
    static_assert(util::is_model_expr_v<RHSNodeType>);

    using left_node_t = LHSNodeType;
    using right_node_t = RHSNodeType;

	using dist_value_t = std::common_type_t<
		typename util::model_expr_traits<LHSNodeType>::dist_value_t,
		typename util::model_expr_traits<RHSNodeType>::dist_value_t
			>;

    GlueNode(const left_node_t& lhs,
             const right_node_t& rhs) noexcept
        : left_node_{lhs}
        , right_node_{rhs}
    {}

    /**
     * Generic traversal function.
     * Recursively traverses left then right.
     */
    template <class EqNodeFunc>
    void traverse(EqNodeFunc&& eq_f)
    {
        left_node_.traverse(eq_f);
        right_node_.traverse(eq_f);
    }

    template <class EqNodeFunc>
    void traverse(EqNodeFunc&& eq_f) const
    {
        left_node_.traverse(eq_f);
        right_node_.traverse(eq_f);
    }

    /**
     * Computes left node joint pdf then right node joint pdf
     * and returns the product of the two.
     */
    template <class PVecType
            , class F = util::identity>
    auto pdf(const PVecType& pvalues,
             F f = F()) const 
    { return left_node_.pdf(pvalues, f) * right_node_.pdf(pvalues, f); }

    /**
     * Computes left node joint log-pdf then right node joint log-pdf
     * and returns the sum of the two.
     */
    template <class PVecType
            , class F = util::identity>
    auto log_pdf(const PVecType& pvalues,
                 F f = F()) const
    { 
        return left_node_.log_pdf(pvalues, f) + 
                right_node_.log_pdf(pvalues, f); 
    }

    /**
     * Up to constant addition, returns ad expression of log pdf
     * of both sides added together.
     */
    template <class VecADVarType>
    auto ad_log_pdf(const VecADVarType& vars,
                    const VecADVarType& cache) const
    {
        return (left_node_.ad_log_pdf(vars, cache) +
                right_node_.ad_log_pdf(vars, cache));
    }

private:
    left_node_t left_node_;
    right_node_t right_node_;
};

} // namespace expr
} // namespace ppl
