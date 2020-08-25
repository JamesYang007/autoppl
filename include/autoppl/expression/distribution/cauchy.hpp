#pragma once
#include <cassert>
#include <random>
#include <fastad_bits/reverse/stat/cauchy.hpp>
#include <autoppl/util/traits/traits.hpp>
#include <autoppl/expression/distribution/dist_utils.hpp>
#include <autoppl/math/density.hpp>
#include <autoppl/math/math.hpp>

#define PPL_CAUCHY_PARAM_SHAPE \
    "Cauchy parameters loc and scale must be either scalar or vector. "

namespace ppl {
namespace expr {
namespace dist {
namespace details {

/**
 * Checks whether loc, scale have proper relative shapes.
 * Must be proper shapes and cannot be matrices.
 */
template <class LocType
        , class ScaleType>
struct cauchy_valid_param_dim
{
    static constexpr bool value =
        util::is_shape_v<LocType> &&
        util::is_shape_v<ScaleType> &&
        !util::is_mat_v<LocType> &&
        !util::is_mat_v<ScaleType>;
};

/**
 * Checks if var, loc, scale have proper relative shapes.
 * Currently, we only allow up to vector dimension (no matrix).
 */
template <class VarType
        , class LocType
        , class ScaleType>
struct cauchy_valid_dim
{
    static constexpr bool value =
        util::is_shape_v<VarType> &&
        (
            (util::is_scl_v<VarType> && 
             util::is_scl_v<LocType> &&
             util::is_scl_v<ScaleType>) ||
            (util::is_vec_v<VarType> && 
             cauchy_valid_param_dim<LocType, ScaleType>::value)        
        );
};

template <class LocType
        , class ScaleType>
inline constexpr bool cauchy_valid_param_dim_v =
    cauchy_valid_param_dim<LocType, ScaleType>::value;

template <class VarType
        , class LocType
        , class ScaleType>
inline constexpr bool cauchy_valid_dim_v =
    cauchy_valid_dim<VarType, LocType, ScaleType>::value;

} // namespace details

/**
 * Cauchy is a generic expression type for the cauchy distribution.
 *
 * If LocType or ScaleType is a vector, then the variable assigned
 * to this distribution must also be a vector.
 *
 * @tparam  LocType     variable expression for the loc.
 *                      Cannot be a matrix shape.
 * @tparam  ScaleType   variable expression for the scale.
 *                      Cannot be a matrix shape.
 */

template <class LocType
        , class ScaleType>
struct Cauchy: util::DistExprBase<Cauchy<LocType, ScaleType>>
{
private:
    using loc_t = LocType;
    using scale_t = ScaleType;

    static_assert(util::is_var_expr_v<loc_t>);
    static_assert(util::is_var_expr_v<scale_t>);
    static_assert(details::cauchy_valid_param_dim_v<loc_t, scale_t>,
                  PPL_DIST_SHAPE_MISMATCH
                  PPL_CAUCHY_PARAM_SHAPE
                  );

public:
    using value_t = util::cont_param_t;
    using base_t = util::DistExprBase<Cauchy<loc_t, scale_t>>; 
    using typename base_t::dist_value_t;

    Cauchy(const loc_t& loc, 
           const scale_t& scale)
        : loc_{loc}, scale_{scale} 
    {}

    //template <class XType>
    //dist_value_t pdf(const XType& x) 
    //{
    //    static_assert(util::is_dist_assignable_v<XType>);
    //    static_assert(details::cauchy_valid_dim_v<XType, loc_t, scale_t>,
    //                  PPL_DIST_SHAPE_MISMATCH);
    //    return math::cauchy_pdf(x.get(), loc_.eval(), scale_.eval());
    //}

    template <class XType>
    dist_value_t log_pdf(const XType& x) 
    {
        static_assert(util::is_dist_assignable_v<XType>);
        static_assert(details::cauchy_valid_dim_v<XType, loc_t, scale_t>,
                      PPL_DIST_SHAPE_MISMATCH);
        return math::cauchy_log_pdf(x.get(), loc_.eval(), scale_.eval());
    }

    template <class XType
            , class PtrPackType>
    auto ad_log_pdf(const XType& x,
                    const PtrPackType& pack) const
    {
        return ad::cauchy_adj_log_pdf(x.ad(pack),
                                      loc_.ad(pack),
                                      scale_.ad(pack));
    }

    template <class PtrPackType>
    void bind(const PtrPackType& pack)
    { 
        static_cast<void>(pack);
        if constexpr (loc_t::has_param) {
            loc_.bind(pack);
        }
        if constexpr (scale_t::has_param) {
            scale_.bind(pack);
        }
    }

    void activate_refcnt() const 
    { 
        loc_.activate_refcnt(); 
        scale_.activate_refcnt();
    }

    template <class XType, class GenType>
    constexpr bool prune(XType&, GenType&) const { return false; }

private:
    loc_t loc_; 
    scale_t scale_;
};

} // namespace dist
} // namespace expr

/**
 * Builds a Cauchy expression only when the parameters
 * are both valid continuous distribution parameter types.
 * See var_expr.hpp for more information.
 */
template <class LocType, class ScaleType
        , class = std::enable_if_t<
            util::is_valid_dist_param_v<LocType> &&
            util::is_valid_dist_param_v<ScaleType>
         > >
inline constexpr auto cauchy(const LocType& loc_expr,
                              const ScaleType& scale_expr)
{
    using loc_t = util::convert_to_param_t<LocType>;
    using scale_t = util::convert_to_param_t<ScaleType>;

    loc_t wrap_loc_expr = loc_expr;
    scale_t wrap_scale_expr = scale_expr;

    return expr::dist::Cauchy(wrap_loc_expr, wrap_scale_expr);
}

} // namespace ppl

#undef PPL_CAUCHY_PARAM_SHAPE
