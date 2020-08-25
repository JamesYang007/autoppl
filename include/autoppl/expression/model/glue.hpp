#pragma once
#include <type_traits>
#include <autoppl/util/traits/model_expr_traits.hpp>

namespace ppl {
namespace expr {
namespace model {

/**
 * This class represents a "node" in a model expression that
 * "glues" two sub-model expressions.
 */
template <class LHSNodeType
        , class RHSNodeType>
struct GlueNode: 
    util::ModelExprBase<GlueNode<LHSNodeType, RHSNodeType>>
{
    static_assert(util::is_model_expr_v<LHSNodeType>);
    static_assert(util::is_model_expr_v<RHSNodeType>);

    using lhs_t = LHSNodeType;
    using rhs_t = RHSNodeType;

	using dist_value_t = std::common_type_t<
		typename util::model_expr_traits<lhs_t>::dist_value_t,
		typename util::model_expr_traits<rhs_t>::dist_value_t
			>;

    GlueNode(const lhs_t& lhs,
             const rhs_t& rhs) noexcept
        : lhs_{lhs}
        , rhs_{rhs}
    {}

    /**
     * Generic traversal function.
     * Recursively traverses left then right.
     */
    template <class EqNodeFunc>
    void traverse(EqNodeFunc&& eq_f)
    {
        lhs_.traverse(eq_f);
        rhs_.traverse(eq_f);
    }

    template <class EqNodeFunc>
    void traverse(EqNodeFunc&& eq_f) const
    {
        lhs_.traverse(eq_f);
        rhs_.traverse(eq_f);
    }

    /**
     * Computes left node joint pdf then right node joint pdf
     * and returns the product of the two.
     */
    auto pdf() { return lhs_.pdf() * rhs_.pdf(); }

    /**
     * Computes left node joint log-pdf then right node joint log-pdf
     * and returns the sum of the two.
     */
    auto log_pdf() { return lhs_.log_pdf() + rhs_.log_pdf(); }

    /**
     * Up to constant addition, returns ad expression of log pdf
     * of both sides added together.
     */
    template <class PtrPackType>
    auto ad_log_pdf(const PtrPackType& pack) const
    {
        return (lhs_.ad_log_pdf(pack) +
                rhs_.ad_log_pdf(pack));
    }

    template <class PtrPackType>
    void bind(const PtrPackType& pack)
    { 
        lhs_.bind(pack);
        rhs_.bind(pack);
    }

    void activate_refcnt() const {
        lhs_.activate_refcnt();
        rhs_.activate_refcnt();
    }

private:
    lhs_t lhs_;
    rhs_t rhs_;
};

} // namespace model
} // namespace expr
} // namespace ppl
