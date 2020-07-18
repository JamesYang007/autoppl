#pragma once
#include <type_traits>
#include <autoppl/util/traits/var_traits.hpp>
#include <autoppl/util/traits/model_expr_traits.hpp>
#include <autoppl/util/traits/dist_expr_traits.hpp>
#include <autoppl/util/functional.hpp>

#define PPL_VAR_DIST_CONT_DISC_MATCH \
    "A continuous variable can only be assigned to a continuous distribution. " \
    "A discrete variable can only be assigned to a discrete distribution. "

namespace ppl {
namespace expr {

/**
 * This class represents a node in the model expression
 * that relates a variable with a distribution.
 * It cannot relate a variable expression in general to a distribution.
 */
template <class VarType
        , class DistType>
struct EqNode: util::ModelExprBase<EqNode<VarType, DistType>>
{
    using var_t = VarType;
    using dist_t = DistType;

    static_assert(util::is_var_v<var_t>);
    static_assert(util::is_dist_expr_v<dist_t>);

    static_assert((util::var_traits<var_t>::is_cont_v &&
                   util::dist_expr_traits<dist_t>::is_cont_v) ||
                  (util::var_traits<var_t>::is_disc_v &&
                   util::dist_expr_traits<dist_t>::is_disc_v),
                  PPL_VAR_DIST_CONT_DISC_MATCH);

    using dist_value_t = typename
        util::dist_expr_traits<dist_t>::dist_value_t;

    EqNode(const var_t& var, 
           const dist_t& dist) noexcept
        : var_{var}
        , dist_{dist}
    {}

    /**
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

    template <class EqNodeFunc>
    void traverse(EqNodeFunc&& eq_f) const
    {
        using this_t = EqNode<VarType, DistType>;
        eq_f(static_cast<const this_t&>(*this));
    }

    /**
     * Compute pdf of underlying distribution with underlying value.
     * Assumes that underlying value has been assigned properly.
     */
    template <class PVecType
            , class F = util::identity>
    auto pdf(const PVecType& pvalues,
             F f = F()) const 
    { return dist_.pdf(get_variable(), pvalues, f); }

    /**
     * Compute log-pdf of underlying distribution with underlying value.
     * Assumes that underlying value has been assigned properly.
     */
    template <class PVecType
            , class F = util::identity>
    auto log_pdf(const PVecType& pvalues,
                 F f = F()) const 
    { return dist_.log_pdf(get_variable(), pvalues, f); }

    /**
     * Generates AD expression for log pdf of underlying distribution.
     * @param   map     mapping of variable IDs to offset in ad_vars
     * @param   ad_vars container of AD variables that correspond to parameters.
     */
    template <class VecADVarType>
    auto ad_log_pdf(const VecADVarType& ad_vars,
                    const VecADVarType& cache) const
    { return dist_.ad_log_pdf(get_variable(), ad_vars, cache); }

    var_t& get_variable() { return var_; }
    const var_t& get_variable() const { return var_; }
    dist_t& get_distribution() { return dist_; }
    const dist_t& get_distribution() const { return dist_; }

private:
    var_t var_; 
    dist_t dist_;
};

} // namespace expr
} // namespace ppl
