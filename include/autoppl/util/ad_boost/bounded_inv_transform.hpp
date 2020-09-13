#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/size_pack.hpp>
#include <fastad_bits/util/value.hpp>
#include <autoppl/util/ad_boost/value.hpp>

namespace ad {
namespace boost {

template <class UCType
        , class LowerType
        , class UpperType
        , class CType>
inline constexpr void bounded_inv_transform(const UCType& uc,
                                            const LowerType& lower,
                                            const UpperType& upper,
                                            CType& c) 
{ 
    using std::exp;
    using Eigen::exp;
    auto auc = util::to_array(uc);
    auto alower = util::to_array(lower);
    auto aupper = util::to_array(upper);
    c = alower + (aupper - alower) / (1. + exp(-auc)); 
}

template <class ExprType
        , class LowerType
        , class UpperType>
struct BoundedInvTransformNode:
    core::ValueAdjView<typename util::expr_traits<ExprType>::value_t,
                   typename util::shape_traits<ExprType>::shape_t>,
    core::ExprBase<BoundedInvTransformNode<ExprType, LowerType, UpperType>>
{
private:
    using expr_t = ExprType;
    using expr_value_t = typename util::expr_traits<expr_t>::value_t;
    using expr_shape_t = typename util::expr_traits<expr_t>::shape_t;
    using lower_t = LowerType;
    using upper_t = UpperType;

public:
    using value_adj_view_t = core::ValueAdjView<expr_value_t, expr_shape_t>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    BoundedInvTransformNode(const expr_t& expr,
                            const lower_t& lower,
                            const upper_t& upper,
                            value_t* c_val,
                            size_t* visit_cnt,
                            size_t refcnt)
        : value_adj_view_t(c_val, nullptr, expr.rows(), expr.cols())
        , expr_{expr}
        , lower_{lower}
        , upper_{upper}
        , scaled_inv_logit_()
        , inv_logit_()
        , v_val_{visit_cnt}
        , refcnt_{refcnt}
    {}

    const var_t& feval()
    {
        auto&& lower = lower_.feval();
        auto&& upper = upper_.feval();
        ++*v_val_;
        if (*v_val_ == 1) {
            auto&& uc_val = expr_.feval();
            bounded_inv_transform(uc_val, lower, upper, this->get());
        }
        *v_val_  = *v_val_ % refcnt_;
        return this->get();
    }

    template <class T>
    void beval(const T& seed)
    {
        auto&& a_val = util::to_array(this->get());
        auto&& a_adj = util::to_array(this->get_adj());
        auto&& a_lower = util::to_array(lower_.get());
        auto&& a_upper = util::to_array(upper_.get());
        
        a_adj = seed;

        scaled_inv_logit_ = (a_val - a_lower);
        auto&& a_scaled_inv_logit = util::to_array(scaled_inv_logit_);

        inv_logit_ = a_scaled_inv_logit / (a_upper - a_lower);
        auto&& a_inv_logit = util::to_array(inv_logit_);

        if constexpr (util::is_scl_v<upper_t>) {
            upper_.beval(sum(a_adj * a_inv_logit));
        } else {
            upper_.beval(a_adj * a_inv_logit);
        }

        if constexpr (util::is_scl_v<lower_t>) {
            lower_.beval(sum(a_adj * (1. - a_inv_logit)));
        } else {
            lower_.beval(a_adj * (1. - a_inv_logit));
        }

        expr_.beval(a_adj * a_scaled_inv_logit * (1. - a_inv_logit));
    }

    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = expr_.bind_cache(begin);
        begin = lower_.bind_cache(begin);
        begin = upper_.bind_cache(begin);
        auto val = begin.val;
        begin.val = this->data();
        begin = this->bind(begin);
        begin.val = val;
        return begin;
    }

    util::SizePack bind_cache_size() const
    {
        return single_bind_cache_size() +
                expr_.bind_cache_size() +
                lower_.bind_cache_size() +
                upper_.bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const { 
        return {0, this->size()}; 
    }

private:
    expr_t expr_;
    lower_t lower_;
    upper_t upper_;
    util::constant_var_t<value_t, shape_t> scaled_inv_logit_;
    util::constant_var_t<value_t, shape_t> inv_logit_;
    size_t* v_val_;
    size_t const refcnt_;
};

template <class ExprType
        , class LowerType
        , class UpperType>
struct LogJBoundedInvTransformNode:
    core::ValueAdjView<typename util::expr_traits<ExprType>::value_t, ad::scl>,
    core::ExprBase<LogJBoundedInvTransformNode<ExprType, LowerType, UpperType>>
{
private:
    using expr_t = ExprType;
    using lower_t = LowerType;
    using upper_t = UpperType;
    using expr_value_t = typename util::expr_traits<expr_t>::value_t;
    using expr_shape_t = typename util::shape_traits<expr_t>::shape_t;
    using lower_shape_t = typename util::shape_traits<lower_t>::shape_t;
    using upper_shape_t = typename util::shape_traits<upper_t>::shape_t;

public:
    using value_adj_view_t = core::ValueAdjView<expr_value_t, ad::scl>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    LogJBoundedInvTransformNode(const expr_t& expr,
                                const lower_t& lower,
                                const upper_t& upper,
                                value_t* c_val)
        : value_adj_view_t(nullptr, nullptr, 1, 1)
        , expr_{expr}
        , lower_{lower}
        , upper_{upper}
        , c_val_(c_val, expr.rows(), expr.cols())
        , scaled_inv_logit_()
    {}

    const var_t& feval()
    {
        using std::log;
        using Eigen::log;

        expr_.feval();
        auto&& a_c_val = util::to_array(c_val_.get());
        auto&& a_lower = util::to_array(lower_.feval());
        auto&& a_upper = util::to_array(upper_.feval());

        scaled_inv_logit_ = a_c_val - a_lower;
        inv_range_ = 1. / (a_upper - a_lower);

        auto&& a_scaled_inv_logit = util::to_array(scaled_inv_logit_);

        // this may be slightly inefficient since this requires extra multiplication,
        // but the benefit is that we can save inverse range, which gets reused a lot in beval
        auto&& a_inv_logit = a_scaled_inv_logit * util::to_array(inv_range_);

        return this->get() = sum(log(a_scaled_inv_logit * (1. - a_inv_logit)));
    }

    void beval(value_t seed)
    {
        auto a_inv_range = util::to_array(inv_range_);
        auto a_scaled_inv_logit = util::to_array(scaled_inv_logit_);

        if constexpr (util::is_scl_v<upper_t> &&
                      util::is_scl_v<lower_t>) {
            upper_.beval(seed * expr_.size() * a_inv_range);
            lower_.beval(seed * expr_.size() * (-a_inv_range));

        } else if constexpr (util::is_scl_v<upper_t>) {
            upper_.beval(seed * a_inv_range.sum());
            lower_.beval(seed * (-a_inv_range));

        } else if constexpr (util::is_scl_v<lower_t>) {
            upper_.beval(seed * a_inv_range);
            lower_.beval(seed * (-a_inv_range.sum()));

        } else {
            assert(upper_.cols() == lower_.cols());
            assert(upper_.rows() == lower_.rows());
            upper_.beval(seed * a_inv_range);
            lower_.beval(seed * -a_inv_range);
        }

        expr_.beval(seed * (1. - 2. * a_scaled_inv_logit * a_inv_range));    
    }

    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = expr_.bind_cache(begin);
        begin = lower_.bind_cache(begin);
        begin = upper_.bind_cache(begin);
        auto adj = begin.adj;
        begin.adj = nullptr;
        begin = this->bind(begin);
        begin.adj = adj;
        return begin;
    }

    util::SizePack bind_cache_size() const
    {
        return single_bind_cache_size() +
                expr_.bind_cache_size() +
                lower_.bind_cache_size() +
                upper_.bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const { 
        return {this->size(), 0}; 
    }

private:
    using view_t = core::ValueView<expr_value_t, expr_shape_t>;
    expr_t expr_;
    lower_t lower_;
    upper_t upper_;
    view_t c_val_;
    util::constant_var_t<expr_value_t, expr_shape_t> scaled_inv_logit_;
    util::constant_var_t<expr_value_t, 
        util::max_shape_t<lower_shape_t, upper_shape_t> > inv_range_;
};

} // namespace boost
} // namespace ad
