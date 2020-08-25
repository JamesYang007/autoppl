#pragma once
#include <cstddef>
#include <cmath>
#include <fastad_bits/util/shape_traits.hpp>
#include <fastad_bits/reverse/core/var_view.hpp>
#include <autoppl/expression/constraint/transformer.hpp>
#include <autoppl/util/ad_boost/bounded_inv_transform.hpp>
#include <autoppl/util/traits/traits.hpp>
#include <autoppl/util/value.hpp>

namespace ppl {
namespace expr {
namespace constraint {

template <class LowerType, class UpperType>
struct Bounded 
{
private:
    using lower_t = LowerType;
    using upper_t = UpperType;

    static_assert(util::is_var_expr_v<lower_t>);
    static_assert(util::is_var_expr_v<upper_t>);
    static_assert(util::is_scl_v<lower_t> ||
                  util::is_scl_v<upper_t> ||
                  std::is_same_v<
                    typename util::shape_traits<lower_t>::shape_t,
                    typename util::shape_traits<upper_t>::shape_t >);

public:

    Bounded(const lower_t& lower, 
            const upper_t& upper)
        : lower_{lower}
        , upper_{upper}
    {}

    /**
     * Transforms from constrained (c) to unconstrained (uc).
     */
    template <class T>
    constexpr void transform(const T& c,
                             T& uc) const
    { 
        if constexpr (std::is_arithmetic_v<T>) {
            uc = std::log((c - lower_.get()) / (upper_.get() - c)); 
        } else {
            auto ca = c.array();
            auto alower = util::to_array(lower_.get());
            auto aupper = util::to_array(upper_.get());
            uc = (ca - alower) / (aupper - ca).log().matrix();
        }
    }

    /**
     * Inverse transforms from unconstrained (uc) to constrained (c).
     */
    template <class T>
    constexpr void inv_transform(const T& uc,
                                 T& c) 
    { ad::boost::bounded_inv_transform(uc, lower_.eval(), upper_.eval(), c); }

    template <class UCViewType
            , class CurrPtrPackType
            , class PtrPackType>
    auto inv_transform_ad(const UCViewType& uc_view, 
                          const CurrPtrPackType& curr_pack,
                          const PtrPackType& pack,
                          size_t refcnt) const
    {
        auto&& lower = lower_.ad(pack);
        auto&& upper = upper_.ad(pack);
        return ad::boost::BoundedInvTransformNode(uc_view, 
                                                  lower,
                                                  upper,
                                                  curr_pack.c_val,
                                                  curr_pack.v_val, 
                                                  refcnt);
    }

    template <class UCViewType
            , class CurrPtrPack
            , class PtrPack>
    auto logj_inv_transform_ad(const UCViewType& uc_view,
                               const CurrPtrPack& curr_pack,
                               const PtrPack& pack) const
    {
        auto lower = lower_.ad(pack);
        auto upper = upper_.ad(pack);
        return ad::boost::LogJBoundedInvTransformNode(uc_view, 
                                                      lower,
                                                      upper,
                                                      curr_pack.c_val);
    }

    void activate_refcnt() const { 
        lower_.activate_refcnt(); 
        upper_.activate_refcnt();
    }

    template <class PtrPack>
    void bind(const PtrPack& pack) { 
        if constexpr (lower_t::has_param) {
            lower_.bind(pack);
        } 
        if constexpr (upper_t::has_param) {
            upper_.bind(pack);
        } 
    }

private:
    lower_t lower_;
    upper_t upper_;
};

// Specialization: Lower (scalar)
template <class ValueType
        , class ShapeType
        , class LowerType
        , class UpperType>
struct Transformer<ValueType, ShapeType, Bounded<LowerType, UpperType>>
{
    using value_t = ValueType;
    using shape_t = ShapeType;
    using var_t = util::var_t<value_t, shape_t>;
    using constraint_t = Bounded<LowerType, UpperType>;
    using uc_view_t = ad::util::shape_to_raw_view_t<value_t, shape_t>;
    using view_t = uc_view_t;

    // only continuous value types can be constrained
    static_assert(util::is_cont_v<value_t>);

    /**
     * Constructs a Transformer object.
     * It represents a lower and upper bounded scalar.
     *
     * @param   c    constraint object must be provided since it cannot be default-constructed
     */
    Transformer(size_t rows, 
                size_t cols,
                constraint_t c)
        : uc_val_(util::make_val<value_t, shape_t>(rows, cols))
        , c_val_(util::make_val<value_t, shape_t>(rows, cols))
        , v_val_(nullptr)
        , constraint_(c)
    {}

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
     * Inverse transform from unconstrained parameters to constrained parameters.
     * This should not have any memory dependency through calling bind.
     * It is expected that uc and c are 1x1 block of Eigen matrix acting as scalar.
     *
     * @param   uc  unconstrained parameter to transform
     * @param   c   constrained parameter to populate
     */
    //template <class UCType, class CType>
    //void inv_transform(const UCType& uc,
    //                   CType& c) const
    //{ constraint_.inv_transform(uc(0,0), c(0,0)); }

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
     * In general, this may need to reuse computed values from inverse transform.
     * User must guarantee that inverse transform AD expression that is bound to the same
     * resources as the return value of this function is evaluated before.
     */
    template <class CurrPtrPack, class PtrPack>
    auto logj_inv_transform_ad(const CurrPtrPack& curr_pack,
                               const PtrPack& pack) const {
        ad::VarView<value_t, shape_t> uc_view(curr_pack.uc_val,     
                                              curr_pack.uc_adj,
                                              rows_uc(),
                                              cols_uc());
        return constraint_.logj_inv_transform_ad(uc_view, curr_pack, pack);
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


template <class LowerType
        , class UpperType>
constexpr inline auto bounded(const LowerType& lower,
                              const UpperType& upper)
{
    using lower_t = util::convert_to_param_t<LowerType>;
    using upper_t = util::convert_to_param_t<UpperType>;
    lower_t wrap_lower = lower;
    upper_t wrap_upper = upper;
    return expr::constraint::Bounded(wrap_lower, wrap_upper);
}

} // namespace ppl

