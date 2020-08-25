#pragma once
#include <fastad_bits/reverse/stat/normal.hpp>
#include <autoppl/util/traits/traits.hpp>
#include <autoppl/expression/distribution/dist_utils.hpp>
#include <autoppl/math/density.hpp>
#include <autoppl/math/math.hpp>

#define PPL_NORMAL_PARAM_SHAPE \
    "Normal distribution mean must either be a scalar or vector. "

namespace ppl {
namespace expr {
namespace dist {
namespace details {

/**
 * Checks case 1 of whether mean, and sigma have proper relative shapes.
 * Case 1: mean, sigma are all scalars.
 */
template <class MeanType
        , class SigmaType>
struct normal_valid_param_dim_case_1
{
    static constexpr bool value =
        util::is_shape_v<MeanType> &&
        util::is_shape_v<SigmaType> &&
        util::is_scl_v<MeanType> &&
        util::is_scl_v<SigmaType>;
};

/**
 * Checks case 2 of whether mean, and sigma have proper relative shapes.
 * Case 2: MeanType is non-matrix.
 */
template <class MeanType
        , class SigmaType>
struct normal_valid_param_dim_case_2
{
    static constexpr bool value =
        util::is_shape_v<MeanType> &&
        util::is_shape_v<SigmaType> &&
        !util::is_mat_v<MeanType>;
};

/**
 * Checks if var, mean, and sigma have proper relative shapes.
 */
template <class VarType
        , class MeanType
        , class SigmaType>
struct normal_valid_dim
{
    static constexpr bool value =
        util::is_shape_v<VarType> &&
        (
            (util::is_scl_v<VarType> &&
                normal_valid_param_dim_case_1<MeanType, SigmaType>::value) ||
            (util::is_vec_v<VarType> && 
                normal_valid_param_dim_case_2<MeanType, SigmaType>::value)        
        );
};

template <class MeanType
        , class SigmaType>
inline constexpr bool normal_valid_param_dim_case_1_v =
    normal_valid_param_dim_case_1<MeanType, SigmaType>::value;

template <class MeanType
        , class SigmaType>
inline constexpr bool normal_valid_param_dim_case_2_v =
    normal_valid_param_dim_case_2<MeanType, SigmaType>::value;

template <class VarType
        , class MeanType
        , class SigmaType>
inline constexpr bool normal_valid_dim_v =
    normal_valid_dim<VarType, MeanType, SigmaType>::value;

} // namespace details

/**
 * Normal is a generic distribution expression representing
 * the normal distribution.
 *
 * If MeanType is a vector, then the variable assigned to this
 * distribution must also be a vector.
 *
 * @tparam  MeanType    variable expression type for the mean.
 *                      Must be either a scalar or vector shape.
 * @tparam  SigmaType   variable expression type for the sigma.
 */

template <class MeanType
        , class SigmaType>
struct Normal: 
    util::DistExprBase<Normal<MeanType, SigmaType>>
{
private:
    using mean_t = MeanType;
    using sigma_t = SigmaType;

    static_assert(util::is_var_expr_v<mean_t>);
    static_assert(util::is_var_expr_v<sigma_t>);
    static_assert(details::normal_valid_param_dim_case_2_v<mean_t, sigma_t>,
                  PPL_DIST_SHAPE_MISMATCH
                  PPL_NORMAL_PARAM_SHAPE
                  );

public:
    using value_t = util::cont_param_t;
    using base_t = util::DistExprBase<Normal<mean_t, sigma_t>>;
    using typename base_t::dist_value_t;

    Normal(const mean_t& mean, 
           const sigma_t& sigma)
        : mean_{mean}, sigma_{sigma} 
    {}

    template <class XType>
    dist_value_t pdf(const XType& x) 
    { 
        static_assert(util::is_dist_assignable_v<XType>);
        static_assert(details::normal_valid_dim_v<XType, mean_t, sigma_t>,
                      PPL_DIST_SHAPE_MISMATCH);
        return math::normal_pdf(x.get(), mean_.eval(), sigma_.eval());
    }

    template <class XType>
    dist_value_t log_pdf(const XType& x) 
    {
        static_assert(util::is_dist_assignable_v<XType>);
        static_assert(details::normal_valid_dim_v<XType, mean_t, sigma_t>,
                      PPL_DIST_SHAPE_MISMATCH);
        return math::normal_log_pdf(x.get(), mean_.eval(), sigma_.eval());
    }
    
    template <class XType
            , class PtrPackType>
    auto ad_log_pdf(const XType& x,
                    const PtrPackType& pack) const
    {
        static_assert(util::is_dist_assignable_v<XType>);
        static_assert(details::normal_valid_dim_v<XType, mean_t, sigma_t>,
                      PPL_DIST_SHAPE_MISMATCH);
        return ad::normal_adj_log_pdf(x.ad(pack),
                                      mean_.ad(pack),
                                      sigma_.ad(pack));
    }

    template <class PtrPackType>
    void bind(const PtrPackType& pack)
    { 
        static_cast<void>(pack);
        if constexpr (mean_t::has_param) {
            mean_.bind(pack);
        }
        if constexpr (sigma_t::has_param) {
            sigma_.bind(pack);
        }
    }

    void activate_refcnt() const 
    { 
        mean_.activate_refcnt(); 
        sigma_.activate_refcnt();
    }

    template <class XType, class GenType>
    bool prune(XType&, GenType&) const { return false; }

private:
    mean_t mean_; 
    sigma_t sigma_;
};

} // namespace dist
} // namespace expr

/**
 * Builds a Normal expression only when the parameters
 * are both valid continuous distribution parameter types.
 * See var_expr.hpp for more information.
 */
template <class MeanType, class SDType
        , class = std::enable_if_t<
            util::is_valid_dist_param_v<MeanType> &&
            util::is_valid_dist_param_v<SDType>
         > >
inline constexpr auto normal(const MeanType& mean_expr,
                             const SDType& sd_expr)
{
    using mean_t = util::convert_to_param_t<MeanType>;
    using sd_t = util::convert_to_param_t<SDType>;

    mean_t wrap_mean_expr = mean_expr;
    sd_t wrap_sd_expr = sd_expr;

    return expr::dist::Normal(wrap_mean_expr, wrap_sd_expr);
}

} // namespace ppl

#undef PPL_NORMAL_PARAM_SHAPE
