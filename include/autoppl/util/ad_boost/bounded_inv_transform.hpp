#pragma once
#include <fastad_bits/reverse/core/expr_base.hpp>
#include <fastad_bits/reverse/core/value_view.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <autoppl/util/value.hpp>

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
    auto auc = ppl::util::to_array(uc);
    auto alower = ppl::util::to_array(lower);
    auto aupper = ppl::util::to_array(upper);
    c = alower + (aupper - alower) / (1. + exp(-auc)); 
}

template <class ExprType
        , class LowerType
        , class UpperType>
struct BoundedInvTransformNode:
    core::ValueView<typename util::expr_traits<ExprType>::value_t,
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
    using value_view_t = core::ValueView<expr_value_t, expr_shape_t>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    BoundedInvTransformNode(const expr_t& expr,
                            const lower_t& lower,
                            const upper_t& upper,
                            value_t* c_val,
                            size_t* visit_cnt,
                            size_t refcnt)
        : value_view_t(c_val, expr.rows(), expr.cols())
        , expr_{expr}
        , lower_{lower}
        , upper_{upper}
        , v_val_{visit_cnt}
        , refcnt_{refcnt}
    {}

    const var_t& feval()
    {
        ++*v_val_;
        if (*v_val_ == 1) {
            auto&& uc_val = expr_.feval();
            auto&& lower = lower_.feval();
            auto&& upper = upper_.feval();
            bounded_inv_transform(uc_val, lower, upper, this->get());
        }
        *v_val_  = *v_val_ % refcnt_;
        return this->get();
    }

    void beval(value_t seed, size_t i, size_t j, util::beval_policy pol)
    {
        if (seed == 0) return;
        value_t scaled_inv_logit = (this->get(i,j) - lower_.get(i,j));
        value_t inv_logit = scaled_inv_logit / (upper_.get(i,j) - lower_.get(i,j));
        upper_.beval(seed * inv_logit, i, j, pol);
        lower_.beval(seed * (1. - inv_logit), i, j, pol);
        expr_.beval(seed * scaled_inv_logit * (1. - inv_logit), i, j, pol);
    }

    value_t* bind(value_t* begin)
    {
        value_t* next = begin;
        if constexpr (!util::is_var_view_v<expr_t>) {
            next = expr_.bind(next);
        }
        if constexpr (!util::is_var_view_v<expr_t>) {
            next = lower_.bind(next);
        }
        if constexpr (!util::is_var_view_v<expr_t>) {
            next = upper_.bind(next);
        }
        return next;
    }

    constexpr size_t bind_size() const
    {
        return single_bind_size() +
                expr_.bind_size() +
                lower_.bind_size() +
                upper_.bind_size();
    }

    constexpr size_t single_bind_size() const { return 0; }

private:
    expr_t expr_;
    lower_t lower_;
    upper_t upper_;
    size_t* v_val_;
    size_t const refcnt_;
};

template <class ExprType
        , class LowerType
        , class UpperType>
struct LogJBoundedInvTransformNode:
    core::ValueView<typename util::expr_traits<ExprType>::value_t, ad::scl>,
    core::ExprBase<LogJBoundedInvTransformNode<ExprType, LowerType, UpperType>>
{
private:
    using expr_t = ExprType;
    using lower_t = LowerType;
    using upper_t = UpperType;
    using expr_value_t = typename util::expr_traits<expr_t>::value_t;
    using expr_shape_t = typename util::shape_traits<expr_t>::shape_t;

public:
    using value_view_t = core::ValueView<expr_value_t, ad::scl>;
    using typename value_view_t::value_t;
    using typename value_view_t::shape_t;
    using typename value_view_t::var_t;
    using value_view_t::bind;

    LogJBoundedInvTransformNode(const expr_t& expr,
                                const lower_t& lower,
                                const upper_t& upper,
                                value_t* c_val)
        : value_view_t(nullptr)
        , expr_{expr}
        , lower_{lower}
        , upper_{upper}
        , c_val_(c_val, expr.rows(), expr.cols())
    {}

    const var_t& feval()
    {
        expr_.feval();
        auto&& lower = lower_.feval();
        auto&& upper = upper_.feval();

        if constexpr (util::is_scl_v<expr_t> &&
                      util::is_scl_v<lower_t> &&
                      util::is_scl_v<upper_t>) {
            value_t scaled_inv_logit = c_val_.get() - lower;
            return this->get() = std::log(scaled_inv_logit * (1 - scaled_inv_logit / (upper - lower)));
        } else { 
            auto alower = ppl::util::to_array(lower_.get());
            auto aupper = ppl::util::to_array(upper_.get());
            auto scaled_inv_logit = ppl::util::to_array(c_val_.get()) - alower;
            return this->get() = (scaled_inv_logit * (1. - scaled_inv_logit / (aupper - alower))).log().sum();
        }
    }

    void beval(value_t seed, size_t, size_t, util::beval_policy pol)
    {
        if (seed == 0) return;
        auto alower = ppl::util::to_array(lower_.get());
        auto aupper = ppl::util::to_array(upper_.get());
        auto ac = ppl::util::to_array(c_val_.get());
        auto inv_range = 1. / (aupper - alower);
        auto scaled_inv_logit = (ac - alower);

        if constexpr (util::is_scl_v<upper_t> &&
                      util::is_scl_v<lower_t>) {
            upper_.beval(seed * expr_.size() * inv_range, 0, 0, pol);
            lower_.beval(seed * expr_.size() * (-inv_range), 0, 0, pol);

        } else if constexpr (util::is_scl_v<upper_t>) {
            upper_.beval(seed * inv_range.sum(), 0, 0, pol);
            for (size_t j = 0; j < lower_.cols(); ++j) {
                for (size_t i = 0; i < lower_.rows(); ++i) {
                    lower_.beval(seed * (-inv_range(i,j)), i, j, pol);
                }
            }

        } else if constexpr (util::is_scl_v<lower_t>) {
            for (size_t j = 0; j < upper_.cols(); ++j) {
                for (size_t i = 0; i < upper_.rows(); ++i) {
                    upper_.beval(seed * inv_range(i,j), i, j, pol);
                }
            }
            lower_.beval(seed * (-inv_range.sum()), 0, 0, pol);

        } else {
            // NOTE: is it allowed to mix back-evaluation like this? 
            assert(upper_.cols() == lower_.cols());
            assert(upper_.rows() == lower_.rows());
            for (size_t j = 0; j < upper_.cols(); ++j) {
                for (size_t i = 0; i < upper_.rows(); ++i) {
                    value_t adj = seed * inv_range(i,j);
                    upper_.beval(adj, i, j, pol);
                    lower_.beval(-adj, i, j, pol);
                }
            }
        }

        if constexpr (util::is_scl_v<expr_t>) {
            expr_.beval(seed * (1. - 2. * scaled_inv_logit * inv_range), 0, 0, pol);    
        } else {
            auto adj_expr = (1. - 2. * scaled_inv_logit * inv_range);
            for (size_t j = 0; j < expr_.cols(); ++j) {
                for (size_t i = 0; i < expr_.rows(); ++i) {
                    value_t adj = adj_expr(i,j);
                    expr_.beval(seed * adj, i, j, pol);
                }
            }
        }
    }

    value_t* bind(value_t* begin)
    {
        value_t* next = begin;
        if constexpr (!util::is_var_view_v<expr_t>) {
            next = expr_.bind(next);
        }
        if constexpr (!util::is_var_view_v<expr_t>) {
            next = lower_.bind(next);
        }
        if constexpr (!util::is_var_view_v<expr_t>) {
            next = upper_.bind(next);
        }
        return value_view_t::bind(next);
    }

    constexpr size_t bind_size() const
    {
        return single_bind_size() +
                expr_.bind_size() +
                lower_.bind_size() +
                upper_.bind_size();
    }

    constexpr size_t single_bind_size() const { return this->size(); }

private:
    using view_t = core::ValueView<expr_value_t, expr_shape_t>;
    expr_t expr_;
    lower_t lower_;
    upper_t upper_;
    view_t c_val_;
};

} // namespace boost
} // namespace ad
