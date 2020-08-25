#pragma once
#include <cstddef>
#include <cmath>
#include <fastad_bits/util/shape_traits.hpp>
#include <fastad_bits/reverse/core/var_view.hpp>
#include <fastad_bits/reverse/core/sum.hpp>
#include <autoppl/util/traits/dist_expr_traits.hpp>
#include <autoppl/expression/constraint/transformer.hpp>
#include <autoppl/util/ad_boost/lower_inv_transform.hpp>
#include <autoppl/util/traits/traits.hpp>
#include <autoppl/util/value.hpp>

namespace ppl {
namespace expr {
namespace constraint {

template <class ExprType>
struct Lower 
{
private:
    using lower_t = ExprType;
    static_assert(util::is_var_expr_v<lower_t>);

public:
    using value_t = typename util::var_expr_traits<lower_t>::value_t;
    using shape_t = typename util::shape_traits<lower_t>::shape_t;

    Lower(const lower_t& lower): lower_{lower} {}

    /**
     * Transforms from constrained (c) to unconstrained (uc).
     * Assumes lower has already been bound and evaluated.
     */
    template <class T>
    constexpr void transform(const T& c,
                             T& uc) const
    { 
        if constexpr (std::is_arithmetic_v<T>) {
            uc = std::log(c - lower_.get()); 
        } else {
            if constexpr (util::is_scl_v<lower_t>) {
                uc = (c.array() - lower_.get()).log().matrix();
            } else {
                uc = (c.array() - lower_.get().array()).log().matrix();            
            }
        }
    }

    /**
     * Inverse transforms from unconstrained (uc) to constrained (c).
     * Does not assume lower has been evaluated, i.e. this call evaluates the constraint.
     * Assumes that user already called bind on constraint expression.
     */
    template <class T>
    constexpr void inv_transform(const T& uc,
                                 T& c) 
    { ad::boost::lower_inv_transform(uc, lower_.eval(), c); }

    template <class UCViewType
            , class CurrPtrPackType
            , class PtrPackType>
    auto inv_transform_ad(const UCViewType& uc_view,
                          const CurrPtrPackType& curr_pack,
                          const PtrPackType& pack,
                          size_t refcnt) const
    {
        auto lower = lower_.ad(pack);
        return ad::boost::LowerInvTransformNode(uc_view, 
                                                lower,
                                                curr_pack.c_val,
                                                curr_pack.v_val, 
                                                refcnt);
    }
    
    template <class UCViewType>
    auto logj_inv_transform_ad(const UCViewType& uc_view) const
    {
        return ad::sum(uc_view);
    }

    void activate_refcnt() const { lower_.activate_refcnt(); }

    template <class PtrPack>
    void bind(const PtrPack& pack) { 
        if constexpr (lower_t::has_param) {
            lower_.bind(pack);
        } 
    }

private:
    lower_t lower_;
};

// Specialization: Lower
template <class ValueType, class ShapeType, class ExprType>
struct Transformer<ValueType, ShapeType, Lower<ExprType>>
{
    using value_t = ValueType;
    using shape_t = ShapeType;
    using var_t = util::var_t<value_t, shape_t>;
    using constraint_t = Lower<ExprType>;
    using uc_view_t = ad::util::shape_to_raw_view_t<value_t, shape_t>;
    using view_t = uc_view_t;

    // only continuous value types can be constrained
    static_assert(util::is_cont_v<value_t>);
    static_assert(util::is_scl_v<ExprType> ||
            std::is_same_v<shape_t,
                typename util::shape_traits<constraint_t>::shape_t>);

    /**
     * Constructs a Transformer object.
     * It represents a lower-bounded scalar.
     *
     * @param   c    constraint object must be provided since it cannot be default-constructed
     */
    Transformer(size_t rows, 
                size_t cols,
                const constraint_t& c)
        : uc_val_(util::make_val<value_t, shape_t>(rows, cols))
        , c_val_(util::make_val<value_t, shape_t>(rows, cols))
        , v_val_(nullptr)
        , constraint_(c)
    {}

    /**
     * This method is currently only used by Param::inv_eval.
     * It should not handle any reference counting.
     * It also assumes all other variables that are referenced 
     * in the constraint expression have already been evaluated.
     */
    void transform() {
        constraint_.transform(util::get(c_val_), util::get(uc_val_));
    }

    /**
     * Inverse transforms from unconstrained parameters to constrained parameters.
     * Only the first visitor of the visit count will invoke the actual transformation.
     * The reference count is used to reset the visit count if 
     * the visit count has reached refcnt.
     */
    void inv_transform(size_t refcnt) { 
        ++*v_val_;
        if (*v_val_ == 1) {
            constraint_.inv_transform(util::get(uc_val_), util::get(c_val_));
        }
        *v_val_ = *v_val_ % refcnt;
    }

    /**
     * Creates an AD expression representing the inverse transform.
     * User must ensure that this gets called exactly refcnt number of times.
     *
     * @param   uc_val      beginning of unconstrained parameter values
     * @param   uc_adj      beginning of unconstrained parameter adjoints
     * @param   c_val       beginning of constrained value region.
     * @param   v_val       beginning of visit count
     * @param   refcnt      total reference count to determine when to loop visit count back to 0
     */
    template <class CurrPtrPack, class PtrPack>
    auto inv_transform_ad(const CurrPtrPack& curr_pack,
                          const PtrPack& pack,
                          size_t refcnt) const {
        ad::VarView<value_t, shape_t> uc_view(curr_pack.uc_val, 
                                              curr_pack.uc_adj, 
                                              rows_uc(), 
                                              cols_uc());
        return constraint_.inv_transform_ad(uc_view, curr_pack, pack, refcnt);
    }

    /**
     * Creates an AD expression representing the log-jacobian of inverse transform.
     * Log-jacobian of inverse transform is identical to the unconstrained variable itself.
     */
    template <class CurrPtrPack, class PtrPack>
    auto logj_inv_transform_ad(const CurrPtrPack& curr_pack,
                               const PtrPack&) const {
        ad::VarView<value_t, shape_t> uc_view(curr_pack.uc_val, 
                                              curr_pack.uc_adj, 
                                              rows_uc(), 
                                              cols_uc());
        return constraint_.logj_inv_transform_ad(uc_view);
    }

    /**
     * Initializes unconstrained values such that unconstrained value is randomly generated determined by dist.
     */
    template <class GenType, class ContDist>
    void init(GenType& gen, ContDist& dist) {
        if constexpr (std::is_same_v<shape_t, scl>) {
            util::get(uc_val_) = dist(gen);
        } else {
            util::get(uc_val_) = var_t::NullaryExpr(rows_uc(), cols_uc(),
                                    [&](size_t, size_t) { return dist(gen); });
        }
    }

    /**
     * Activates reference count for the constraint expression.
     * Since the constraint expression will only be evaluated once per model evaluation,
     * we chose to only activate constraint when the current reference count is 1,
     * i.e. only the first reference to a parameter gets to activate the constraint.
     */
    void activate_refcnt(size_t curr_refcnt) const {
        if (curr_refcnt == 1) constraint_.activate_refcnt();
    }

    var_t& get_c() { return util::get(c_val_); }
    const var_t& get_c() const { return util::get(c_val_); }

    /**
     * Returns the dimension information for the viewers of unconstrained
     * and constrained parameters.
     */
    constexpr size_t size_uc() const { return util::size(uc_val_); }
    constexpr size_t rows_uc() const { return util::rows(uc_val_); }
    constexpr size_t cols_uc() const { return util::cols(uc_val_); }
    constexpr size_t size_c() const { return util::size(c_val_); }
    constexpr size_t rows_c() const { return util::rows(c_val_); }
    constexpr size_t cols_c() const { return util::cols(c_val_); }

    /**
     * Returns the number of elements required to bind and compute 
     * unconstrained, constrained parameters and visit count.
     */
    constexpr size_t bind_size_uc() const { return size_uc(); }
    constexpr size_t bind_size_c() const { return size_c(); }
    constexpr size_t bind_size_v() const { return 1; }

    /**
     * Binds unconstrained viewer to unconstrained region,
     * constrained viewer to constrained region,
     * and internal visit count to visit count region.
     */
    template <class CurrPtrPack, class PtrPack>
    void bind(const CurrPtrPack& curr_pack,
              const PtrPack& pack) 
    {
        util::bind(uc_val_, curr_pack.uc_val, rows_uc(), cols_uc());
        util::bind(c_val_, curr_pack.c_val, rows_c(), cols_c());
        util::bind(v_val_, curr_pack.v_val, 1, 1);
        constraint_.bind(pack);
    }

private:
    uc_view_t uc_val_;
    view_t c_val_;
    size_t* v_val_;
    constraint_t constraint_;
};

} // namespace constraint
} // namespace expr

template <class LowerType>
constexpr inline auto lower(const LowerType& expr)
{
    using lower_t = util::convert_to_param_t<LowerType>;
    lower_t wrap_expr = expr;
    return expr::constraint::Lower(wrap_expr);
}

} // namespace ppl

