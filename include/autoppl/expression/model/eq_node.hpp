#pragma once
#include <algorithm>
#include <type_traits>
#include <functional>
#include <fastad>
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

    template <class EqNodeFunc>
    void traverse(EqNodeFunc&& eq_f) const
    {
        using this_t = EqNode<VarType, DistType>;
        eq_f(static_cast<const this_t&>(*this));
    }

    /*
     * Compute pdf of underlying distribution with underlying value.
     * Assumes that underlying value has been assigned properly.
     */
    dist_value_t pdf() const {
        return dist_.pdf(get_variable());
    }

    /*
     * Compute log-pdf of underlying distribution with underlying value.
     * Assumes that underlying value has been assigned properly.
     */
    dist_value_t log_pdf() const {
        return dist_.log_pdf(get_variable());
    }

    template <class VecRefType, class VecADVarType>
    auto ad_log_pdf(const VecRefType& keys,
                    const VecADVarType& vars) const
    {
        // if parameter, find the corresponding variable
        // in vars and return the AD log-pdf with this variable.
        if constexpr (util::is_param_v<var_t>) {
            const void* addr = &orig_var_ref_.get();
            auto it = std::find(keys.begin(), keys.end(), addr);
            assert(it != keys.end());
            size_t idx = std::distance(keys.begin(), it);
            return dist_.ad_log_pdf(vars[idx], keys, vars);
        } 

        // if data, return sum of log_pdf where each element
        // is a constant AD node containing each value of data.
        // note: data is not copied at any point.
        else if constexpr (util::is_data_v<var_t>) {
            const auto& var = this->get_variable();
            size_t idx = 0;
            const size_t size = var.size();
            return ad::sum(var.begin(), var.end(), 
                    [&, idx, size](auto value) mutable {
                        idx = idx % size; // may be important since mutable
                        auto&& expr = dist_.ad_log_pdf(
                                ad::constant(value), keys, vars, idx);
                        ++idx;
                        return expr;
                    });
        }
    }

    auto& get_variable() { return orig_var_ref_.get(); }
    const auto& get_variable() const { return orig_var_ref_.get(); }
    const auto& get_distribution() const { return dist_; }

private:
    using var_ref_t = std::reference_wrapper<var_t>;    
    var_ref_t orig_var_ref_;      // reference of the original var since 
                                  // any configuration may be changed until right before update 
    dist_t dist_;                 // distribution associated with var
};

} // namespace expr
} // namespace ppl
