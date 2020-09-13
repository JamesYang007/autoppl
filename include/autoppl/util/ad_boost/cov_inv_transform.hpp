#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_adj_view.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <fastad_bits/util/size_pack.hpp>
#include <fastad_bits/util/value.hpp>

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
    core::ValueAdjView<typename util::expr_traits<ExprType>::value_t, ad::mat>,
    core::ExprBase<CovInvTransformNode<ExprType>>
{
private:
    using expr_t = ExprType;
    using expr_value_t = typename util::expr_traits<expr_t>::value_t;
    using expr_shape_t = typename util::shape_traits<expr_t>::shape_t;

    static_assert(util::is_vec_v<expr_t>);

public:
    using value_adj_view_t = core::ValueAdjView<expr_value_t, ad::mat>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    CovInvTransformNode(const expr_t& expr,
                        value_t* lower,
                        value_t* val,
                        size_t rows,
                        size_t* visit_cnt,
                        size_t refcnt)
        : value_adj_view_t(val, nullptr, rows, rows)
        , expr_{expr}
        , lower_(lower, rows, rows)
        , adj_()
        , flattened_adj_(expr.size())
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

    template <class T>
    void beval(const T& seed)
    {
        auto&& a_adj = util::to_array(this->get_adj());
        auto&& a_flattened_adj = util::to_array(flattened_adj_);

        a_adj = seed;
        adj_ = (this->get_adj().transpose() + this->get_adj()) * lower_;
        adj_.diagonal().array() *= lower_.diagonal().array();

        size_t k = 0;
        for (size_t j = 0; j < this->cols(); ++j) {
            for (size_t i = j; i < this->rows(); ++i, ++k) {
                flattened_adj_(k) = adj_(i,j); 
            }
        }

        expr_.beval(a_flattened_adj);
    }

    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = expr_.bind_cache(begin);
        auto val = begin.val;
        begin.val = this->data();
        begin = this->bind(begin);
        begin.val = val;
        return begin;
    }

    util::SizePack bind_cache_size() const
    {
        return single_bind_cache_size() + 
                expr_.bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const { 
        return {0, this->size()}; 
    }

private:
    using mat_view_t = util::shape_to_raw_view_t<value_t, shape_t>;
    expr_t expr_;
    mat_view_t lower_;
    util::constant_var_t<value_t, shape_t> adj_;
    util::constant_var_t<value_t, expr_shape_t> flattened_adj_;
    size_t* v_val_;
    size_t const refcnt_;
};

template <class ExprType>
struct LogJCovInvTransformNode:
    core::ValueAdjView<typename util::expr_traits<ExprType>::value_t, ad::scl>,
    core::ExprBase<LogJCovInvTransformNode<ExprType>>
{
private:
    using expr_t = ExprType;
    using expr_value_t = typename util::expr_traits<expr_t>::value_t;
    using expr_shape_t = typename util::shape_traits<expr_t>::shape_t;

    static_assert(util::is_vec_v<expr_t>);

public:
    using value_adj_view_t = core::ValueAdjView<expr_value_t, ad::scl>;
    using typename value_adj_view_t::value_t;
    using typename value_adj_view_t::shape_t;
    using typename value_adj_view_t::var_t;
    using typename value_adj_view_t::ptr_pack_t;

    LogJCovInvTransformNode(const expr_t& expr,
                            size_t rows)
        : value_adj_view_t(nullptr, nullptr, 1, 1)
        , expr_{expr}
        , rows_{rows}
        , flattened_adj_(expr.size())
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

    void beval(value_t seed)
    {
        size_t weight = rows_ + 1;
        size_t incr = rows_;
        size_t pos = 0;
        flattened_adj_.setZero();
        for (size_t k = 0; k < rows_; ++k, --weight, --incr) {
            flattened_adj_(pos) = weight;
            pos += incr;
        }
        expr_.beval(seed * flattened_adj_.array());
    }

    ptr_pack_t bind_cache(ptr_pack_t begin)
    {
        begin = expr_.bind_cache(begin);
        auto adj = begin.adj;
        begin.adj = nullptr;
        begin = this->bind(begin);
        begin.adj = adj;
        return begin;
    }

    util::SizePack bind_cache_size() const
    {
        return single_bind_cache_size() +
                expr_.bind_cache_size();
    }

    util::SizePack single_bind_cache_size() const { 
        return {this->size(), 0}; 
    }

private:
    expr_t expr_;
    size_t const rows_;
    util::constant_var_t<value_t, expr_shape_t> flattened_adj_;
};

} // namespace boost
} // namespace ad
