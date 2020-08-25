#pragma once
#include <autoppl/util/traits/traits.hpp>

#define PPL_VAR_DIST_CONT_DISC_MATCH \
    "A continuous variable can only be assigned to a continuous distribution. " \
    "A discrete variable can only be assigned to a discrete distribution. "

namespace ppl {
namespace expr {
namespace model {

/**
 * This class represents a node in the model expression
 * that relates a variable with a distribution.
 * It cannot relate a variable expression in general to a distribution.
 */
template <class VarType
        , class DistType>
struct BarEqNode: util::ModelExprBase<BarEqNode<VarType, DistType>>
{
    using var_t = VarType;
    using dist_t = DistType;

    static_assert(util::is_dist_assignable_v<var_t>);
    static_assert(util::is_dist_expr_v<dist_t>);

    static_assert((util::var_traits<var_t>::is_cont_v &&
                   util::dist_expr_traits<dist_t>::is_cont_v) ||
                  (util::var_traits<var_t>::is_disc_v &&
                   util::dist_expr_traits<dist_t>::is_disc_v),
                  PPL_VAR_DIST_CONT_DISC_MATCH);

    using dist_value_t = typename
        util::dist_expr_traits<dist_t>::dist_value_t;

    BarEqNode(const var_t& var, 
              const dist_t& dist) noexcept
        : var_{var}
        , dist_{dist}
    {}

    /**
     * Generic traversal function.
     * Assumes that eq_f is a function that takes in 1 parameter,
     * which is simply the current object.
     */
    template <class BarEqNodeFunc>
    void traverse(BarEqNodeFunc&& eq_f)
    {
        using this_t = BarEqNode<VarType, DistType>;
        eq_f(static_cast<this_t&>(*this));
    }

    template <class BarEqNodeFunc>
    void traverse(BarEqNodeFunc&& eq_f) const
    {
        using this_t = BarEqNode<VarType, DistType>;
        eq_f(static_cast<const this_t&>(*this));
    }

    auto pdf() { 
        var_.eval();
        return dist_.pdf(var_); 
    }
    
    auto log_pdf() { 
        var_.eval();
        return dist_.log_pdf(var_); 
    }

    template <class PtrPackType>
    auto ad_log_pdf(const PtrPackType& pack) const
    { 
        if constexpr (util::is_param_v<var_t>) {
            return dist_.ad_log_pdf(var_, pack) +
                    var_.logj_ad(pack); 
        } else {
            return dist_.ad_log_pdf(var_, pack);
        }
    }

    template <class PtrPackType>
    void bind(const PtrPackType& pack)
    { 
        if constexpr (var_t::has_param) {
            var_.bind(pack);
        }
        dist_.bind(pack);
    }

    void activate_refcnt() const {
        var_.activate_refcnt();
        dist_.activate_refcnt();
    }

    var_t& get_variable() { return var_; }
    const var_t& get_variable() const { return var_; }
    dist_t& get_distribution() { return dist_; }
    const dist_t& get_distribution() const { return dist_; }

private:
    var_t var_; 
    dist_t dist_;
};

} // namespace model
} // namespace expr
} // namespace ppl
