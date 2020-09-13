#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/size_pack.hpp>
#include <fastad_bits/util/value.hpp>
#include <autoppl/util/ad_boost/value.hpp>

namespace ad {
namespace boost {

template <class UCType, class LowerType, class CType>
inline constexpr void lower_inv_transform(const UCType& uc,
                                          const LowerType& lower,
                                          CType& c) 
{ 
    using uc_t = UCType;
    using c_t = std::decay_t<CType>;
    if constexpr (std::is_arithmetic_v<uc_t> &&
                  std::is_arithmetic_v<c_t>) {
        c = std::exp(uc) + lower; 
    } else {
        if constexpr (std::is_arithmetic_v<LowerType>) {
            c = (uc.array().exp() + lower).matrix();
        } else {
            c = (uc.array().exp() + lower.array()).matrix();
        }
    }
}

template <class ExprType
        , class LowerType>
struct LowerInvTransformNode:
    core::ValueAdjView<typename util::expr_traits<ExprType>::value_t, 
                       typename util::expr_traits<ExprType>::shape_t>,
    core::ExprBase<LowerInvTransformNode<ExprType, LowerType>>
{
private:
    using expr_t = ExprType;
    using expr_value_t = typename util::expr_traits<expr_t>::value_t;
    using expr_shape_t = typename util::shape_traits<expr_t>::shape_t;
    using lower_t = LowerType;

    static_assert(util::is_scl_v<lower_t> ||
                  std::is_same_v<
                    typename util::shape_traits<expr_t>::shape_t,
                    typename util::shape_traits<lower_t>::shape_t
                    >);

public:
    using value_adj_view_t = core::ValueAdjView<expr_value_t, expr_shape_t>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    LowerInvTransformNode(const expr_t& expr,
                          const lower_t& lower,
                          value_t* c_val,
                          size_t* visit_cnt,
                          size_t refcnt)
        : value_adj_view_t(c_val, nullptr, expr.rows(), expr.cols())
        , expr_{expr}
        , lower_{lower}
        , v_val_{visit_cnt}
        , refcnt_{refcnt}
    {}

    const var_t& feval()
    {
        auto&& lower = lower_.feval();
        ++*v_val_;
        if (*v_val_ == 1) {
            auto&& uc_val = expr_.feval();
            lower_inv_transform(uc_val, lower, this->get());
        }
        *v_val_ = *v_val_ % refcnt_;
        return this->get();
    }

    template <class T>
    void beval(const T& seed)
    {
        auto&& a_val = util::to_array(this->get());
        auto&& a_adj = util::to_array(this->get_adj());
        auto&& a_lower = util::to_array(lower_.get());
        a_adj = seed;
        if constexpr (util::is_scl_v<lower_t>) {
            lower_.beval(sum(a_adj));
        } else {
            lower_.beval(a_adj);
        }
        expr_.beval(a_adj * (a_val - a_lower));
    }

    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = expr_.bind_cache(begin);
        begin = lower_.bind_cache(begin);
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
                lower_.bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const { 
        return {0,this->size()}; 
    }

private:
    expr_t expr_;
    lower_t lower_;
    size_t* v_val_;
    size_t const refcnt_;
};

} // namespace boost
} // namespace ad
