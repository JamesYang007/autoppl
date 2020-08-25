#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_view.hpp>
#include <fastad_bits/util/type_traits.hpp>

namespace ad {
namespace boost {

template <class LowerType, class UCType, class CType>
inline constexpr void cov_inv_transform(LowerType& lower,
                                        const UCType& uc,
                                        CType& c)
{
    size_t k = 0;
    for (int j = 0; j < lower.cols(); ++j) {
        lower(j,j) = std::exp(uc(k));
        ++k;
        for (int i = j+1; i < lower.rows(); ++i, ++k) {
            lower(i,j) = uc(k); 
        }
    }
    c = lower * lower.transpose();
}

template <class ExprType>
struct CovInvTransformNode:
    core::ValueView<typename util::expr_traits<ExprType>::value_t, ad::mat>,
    core::ExprBase<CovInvTransformNode<ExprType>>
{
private:
    using expr_t = ExprType;
    using expr_value_t = typename 
        util::expr_traits<expr_t>::value_t;

    static_assert(util::is_vec_v<expr_t>);

public:
    using value_view_t = core::ValueView<expr_value_t, ad::mat>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    CovInvTransformNode(const expr_t& expr,
                        value_t* lower,
                        value_t* val,
                        size_t rows,
                        size_t* visit_cnt,
                        size_t refcnt)
        : value_view_t(val, rows, rows)
        , expr_{expr}
        , lower_(lower, rows, rows)
        , v_val_{visit_cnt}
        , refcnt_{refcnt}
    {}

    const var_t& feval()
    {
        ++*v_val_;
        if (*v_val_ == 1) {
            auto&& uc_val_ = expr_.feval();
            cov_inv_transform(lower_, uc_val_, this->get());
        }
        *v_val_  = *v_val_ % refcnt_;
        return this->get();
    }

    void beval(value_t seed, size_t i, size_t j, util::beval_policy pol)
    {
        if (seed == 0) return;

        auto& z = lower_;
        size_t min = std::min(i,j);

        size_t i_pos = i;
        size_t j_pos = j;
        int incr = static_cast<int>(this->rows()) - 1;
        for (size_t k = 0; k < min; ++k, --incr) {
            expr_.beval(seed * z(j,k), i_pos, 0, pol);
            expr_.beval(seed * z(i,k), j_pos, 0, pol);
            i_pos += incr;
            j_pos += incr;
        }

        auto last_adj = [&](size_t i, size_t j) {
            return (i == min) ? 
                z(i,min) * z(j,min) : z(j,min);
        };

        expr_.beval(seed * last_adj(i, j), i_pos, 0, pol);
        expr_.beval(seed * last_adj(j, i), j_pos, 0, pol);
    }

    value_t* bind(value_t* begin)
    {
        value_t* next = begin;
        if constexpr (!util::is_var_view_v<expr_t>) {
            next = expr_.bind(next);
        }
        return next;
    }

    constexpr size_t bind_size() const
    {
        return single_bind_size() +
                expr_.bind_size();
    }

    constexpr size_t single_bind_size() const { return 0; }

private:
    using mat_view_t = util::shape_to_raw_view_t<value_t, shape_t>;
    expr_t expr_;
    mat_view_t lower_;
    size_t* v_val_;
    size_t const refcnt_;
};

template <class ExprType>
struct LogJCovInvTransformNode:
    core::ValueView<typename util::expr_traits<ExprType>::value_t, ad::scl>,
    core::ExprBase<LogJCovInvTransformNode<ExprType>>
{
private:
    using expr_t = ExprType;
    using expr_value_t = typename 
        util::expr_traits<expr_t>::value_t;

    static_assert(util::is_vec_v<expr_t>);

public:
    using value_view_t = core::ValueView<expr_value_t, ad::scl>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    LogJCovInvTransformNode(const expr_t& expr,
                            size_t rows)
        : value_view_t(nullptr)
        , expr_{expr}
        , rows_{rows}
    {}

    const var_t& feval()
    {
        auto&& expr = expr_.feval();
        size_t weight = rows_ + 1;
        size_t incr = rows_;
        size_t pos = 0;
        this->zero();   // REALLY important
        for (size_t k = 0; k < rows_; ++k, --weight, --incr) {
            this->get() += weight * expr(pos);
            pos += incr;
        }
        return this->get();
    }

    void beval(value_t seed, size_t, size_t, util::beval_policy pol)
    {
        if (seed == 0) return;
        
        size_t weight = rows_ + 1;
        size_t incr = rows_;
        size_t pos = 0;
        for (size_t k = 0; k < rows_; ++k, --weight, --incr) {
            expr_.beval(seed * weight, pos, 0, pol);
            pos += incr;
        }
    }

    value_t* bind(value_t* begin)
    {
        value_t* next = begin;
        if constexpr (!util::is_var_view_v<expr_t>) {
            next = expr_.bind(next);
        }
        return value_view_t::bind(next);
    }

    constexpr size_t bind_size() const
    {
        return single_bind_size() +
                expr_.bind_size();
    }

    constexpr size_t single_bind_size() const { return this->size(); }

private:
    expr_t expr_;
    size_t const rows_;
};

} // namespace boost
} // namespace ad
