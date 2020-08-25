#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_view.hpp>
#include <fastad_bits/util/type_traits.hpp>

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
    core::ValueView<typename util::expr_traits<ExprType>::value_t, 
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
    using value_view_t = core::ValueView<expr_value_t, expr_shape_t>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    LowerInvTransformNode(const expr_t& expr,
                          const lower_t& lower,
                          value_t* c_val,
                          size_t* visit_cnt,
                          size_t refcnt)
        : value_view_t(c_val, expr.rows(), expr.cols())
        , expr_{expr}
        , lower_{lower}
        , v_val_{visit_cnt}
        , refcnt_{refcnt}
    {}

    const var_t& feval()
    {
        ++*v_val_;
        if (*v_val_ == 1) {
            auto&& uc_val = expr_.feval();
            auto&& lower = lower_.feval();
            lower_inv_transform(uc_val, lower, this->get());
        }
        *v_val_ = *v_val_ % refcnt_;
        return this->get();
    }

    void beval(value_t seed, size_t i, size_t j, util::beval_policy pol)
    {
        if (seed == 0) return;
        lower_.beval(seed, i, j, pol);
        expr_.beval(seed * (this->get(i,j) - lower_.get(i,j)), i, j, pol);
    }

    value_t* bind(value_t* begin)
    {
        value_t* next = begin;
        if constexpr (!util::is_var_view_v<expr_t>) {
            next = expr_.bind(next);
        }
        if constexpr (!util::is_var_view_v<lower_t>) {
            next = lower_.bind(next);
        }
        return next;
    }

    constexpr size_t bind_size() const
    {
        return single_bind_size() +
                expr_.bind_size() +
                lower_.bind_size();
    }

    constexpr size_t single_bind_size() const { return 0; }

private:
    expr_t expr_;
    lower_t lower_;
    size_t* v_val_;
    size_t const refcnt_;
};

} // namespace boost
} // namespace ad
