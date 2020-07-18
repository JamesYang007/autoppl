#pragma once
#include <cmath>
#include <cassert>
#include <fastad_bits/node.hpp>
#include <fastad_bits/ifelse.hpp>
#include <fastad_bits/pow.hpp>
#include <fastad_bits/math.hpp>
#include <autoppl/util/traits/var_expr_traits.hpp>
#include <autoppl/util/traits/dist_expr_traits.hpp>
#include <autoppl/util/iterator/counting_iterator.hpp>
#include <autoppl/util/functional.hpp>
#include <autoppl/expression/distribution/dist_utils.hpp>
#include <autoppl/math/density.hpp>
#include <autoppl/math/math.hpp>

#define PPL_NORMAL_PARAM_SHAPE \
    "Normal distribution mean and sd must either be a scalar or vector " \
    "Currently, general covariance matrix is not supported. "

namespace ppl {
namespace expr {
namespace details {

/**
 * Checks case 1 of whether mean, and sd have proper relative shapes.
 * Case 1: mean, sd are all scalars.
 */
template <class MeanType
        , class SDType>
struct normal_valid_param_dim_case_1
{
    static constexpr bool value =
        util::is_shape_v<MeanType> &&
        util::is_shape_v<SDType> &&
        util::is_scl_v<MeanType> &&
        util::is_scl_v<SDType>;
};

/**
 * Checks case 2 of whether mean, and sd have proper relative shapes.
 * Case 2: MeanType is non-matrix and SDType is a scalar.
 */
template <class MeanType
        , class SDType>
struct normal_valid_param_dim_case_2
{
    static constexpr bool value =
        util::is_shape_v<MeanType> &&
        util::is_shape_v<SDType> &&
        !util::is_mat_v<MeanType> &&
        !util::is_mat_v<SDType>;
};

/**
 * Checks if var, mean, and sd have proper relative shapes.
 * Currently, we only allow up to vector dimension (no matrix).
 */
template <class VarType
        , class MeanType
        , class SDType>
struct normal_valid_dim
{
    static constexpr bool value =
        util::is_shape_v<VarType> &&
        (
            (util::is_scl_v<VarType> &&
                normal_valid_param_dim_case_1<MeanType, SDType>::value) ||
            (util::is_vec_v<VarType> && 
                normal_valid_param_dim_case_2<MeanType, SDType>::value)        
        );
};

template <class MeanType
        , class SDType>
inline constexpr bool normal_valid_param_dim_case_1_v =
    normal_valid_param_dim_case_1<MeanType, SDType>::value;

template <class MeanType
        , class SDType>
inline constexpr bool normal_valid_param_dim_case_2_v =
    normal_valid_param_dim_case_2<MeanType, SDType>::value;

template <class VarType
        , class MeanType
        , class SDType>
inline constexpr bool normal_valid_dim_v =
    normal_valid_dim<VarType, MeanType, SDType>::value;

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
 * @tparam  SDType      variable expression type for the standard deviation.
 *                      Must be a scalar.
 */

template <class MeanType
        , class SDType>
struct Normal: 
    util::DistExprBase<Normal<MeanType, SDType>>
{
    static_assert(util::is_var_expr_v<MeanType>);
    static_assert(util::is_var_expr_v<SDType>);
    static_assert(details::normal_valid_param_dim_case_2_v<MeanType, SDType>,
                  PPL_DIST_SHAPE_MISMATCH
                  PPL_NORMAL_PARAM_SHAPE
                  );

    using value_t = util::cont_param_t;
    using base_t = util::DistExprBase<Normal<MeanType, SDType>>;
    using index_t = uint32_t;
    using typename base_t::dist_value_t;

    Normal(const MeanType& mean, 
           const SDType& sd)
        : mean_{mean}, sd_{sd} 
    {}

    template <class VarType
            , class PVecType
            , class F = util::identity>
    dist_value_t pdf(const VarType& x,
                     const PVecType& pvalues,
                     F f = F()) const 
    { 
        static_assert(util::is_var_v<VarType>);
        static_assert(details::normal_valid_dim_v<VarType, MeanType, SDType>,
                      PPL_DIST_SHAPE_MISMATCH);
        return pdf_indep([&](size_t i) {
                            return math::normal_pdf(
                                    x.value(pvalues, i, f), 
                                    mean_.value(pvalues, i, f), 
                                    sd_.value(pvalues, i, f));
                          }, x.size());
    }

    template <class VarType
            , class PVecType
            , class F = util::identity>
    dist_value_t log_pdf(const VarType& x,
                         const PVecType& pvalues,
                         F f = F()) const 
    {
        static_assert(util::is_var_v<VarType>);
        static_assert(details::normal_valid_dim_v<VarType, MeanType, SDType>,
                      PPL_DIST_SHAPE_MISMATCH);
        return log_pdf_indep([&](size_t i) {
                                return math::normal_log_pdf(
                                        x.value(pvalues, i, f), 
                                        mean_.value(pvalues, i, f), 
                                        sd_.value(pvalues, i, f));
                              }, x.size());
    }
    
    /** 
     * Up to constant addition, returns AD expression of log pdf.
     */
    template <class VarType, class VecADVarType>
    auto ad_log_pdf(const VarType& x,
                    const VecADVarType& ad_vars,
                    const VecADVarType& cache) const
    {
        static_assert(util::is_var_v<VarType>);
        static_assert(details::normal_valid_dim_v<VarType, MeanType, SDType>,
                      PPL_DIST_SHAPE_MISMATCH);

        // Case 1: x -> scalar, mean -> scalar, sd -> scalar
        if constexpr (util::is_scl_v<VarType> &&
                      util::is_scl_v<MeanType> &&
                      util::is_scl_v<SDType>) 
        {
            auto&& ad_x = x.to_ad(ad_vars, cache);
            auto&& ad_mean = mean_.to_ad(ad_vars, cache);
            auto&& ad_sd = sd_.to_ad(ad_vars, cache);

            // Subcase 1: sd -> has no param
            // don't cache sd to precompute 
            if constexpr (!SDType::has_param) {
                return ad::if_else(
                        ad_sd > ad::constant(0.),
                        ( (ad::constant(-0.5) / ad::pow<2>(ad_sd)) *
                         ad::pow<2>(ad_x - ad_mean) )
                        - ad::log(ad_sd),
                        ad::constant(math::neg_inf<dist_value_t>)
                       ); 
            }

            // Subcase 2: x -> has param or mean -> has param, sd -> has param
            // don't cache mean to minimize expression size
            else if constexpr (VarType::has_param || MeanType::has_param) {
                return (cache[offset_] = ad_sd,
                        ad::if_else(
                        cache[offset_] > ad::constant(0.),
                        (ad::constant(-0.5) *
                        ad::pow<2>( (ad_x - ad_mean) / cache[offset_] ))
                        - ad::log(cache[offset_]),
                        ad::constant(math::neg_inf<dist_value_t>)
                       ) ); 
            }

            // Subcase 3: x-> has no param, mean -> has no param, sd -> has param
            // don't cache mean to precompute
            else {
                return (cache[offset_] = ad_sd,
                        ad::if_else(
                        cache[offset_] > ad::constant(0.),
                        ( ad::constant(-0.5) * ad::pow<2>(ad_x - ad_mean) ) 
                        / ad::pow<2>(cache[offset_]) 
                        - ad::log(cache[offset_]),
                        ad::constant(math::neg_inf<dist_value_t>)
                       ) ); 
            }
        }

        // Case 2: x -> vec, mean -> scalar, sd -> scalar
        else if constexpr (util::is_vec_v<VarType> &&
                           util::is_scl_v<MeanType> &&
                           util::is_scl_v<SDType>)
        {
            size_t x_size = x.size();
            auto&& ad_mean = mean_.to_ad(ad_vars, cache);
            auto&& ad_sd = sd_.to_ad(ad_vars, cache);

            // Subcase 1: x -> has param
            // cache mean since it is more beneficial to cache when sum is large
            // and it is not possible precompute further with mean when it has no param
            if constexpr (VarType::has_param) {

                // Subsubcase 1: sd has param
                if constexpr (SDType::has_param) {
                    return (cache[offset_] = ad_mean,
                            cache[offset_+1] = ad_sd,
                            ad::if_else(
                            cache[offset_+1] > ad::constant(0.),
                            (ad::constant(-0.5) / ad::pow<2>(cache[offset_+1]))
                            * ad::sum(util::counting_iterator<size_t>(0),
                                      util::counting_iterator<size_t>(x_size),
                                      [&](size_t i) { 
                                        return ad::pow<2>(x.to_ad(ad_vars, cache, i) - cache[offset_]); 
                                        })
                            - (ad::constant<dist_value_t>(x_size) * ad::log(cache[offset_+1])),
                            ad::constant(math::neg_inf<dist_value_t>)
                           ) ); 
                }

                // Subsubcase 2: sd has no param
                // don't cache to precompute
                else {
                    return (cache[offset_] = ad_mean,
                            ad::if_else(
                            ad_sd > ad::constant(0.),
                            (ad::constant(-0.5) / ad::pow<2>(ad_sd))
                            * ad::sum(util::counting_iterator<size_t>(0),
                                      util::counting_iterator<size_t>(x_size),
                                      [&](size_t i) { 
                                        return ad::pow<2>(x.to_ad(ad_vars, cache, i) - cache[offset_]); 
                                        })
                            - (ad::constant<dist_value_t>(x_size) * ad::log(ad_sd)),
                            ad::constant(math::neg_inf<dist_value_t>)
                           ) ); 
                }

            }

            // Subcase 2: x -> has no param
            // Note: this is HUGE optimization here
            else {

                auto sample_mean = ad::sum(util::counting_iterator<size_t>(0),
                                           util::counting_iterator<size_t>(x_size),
                                           [&](size_t i) { 
                                               return x.to_ad(ad_vars, cache, i); 
                                           }) / ad::constant<dist_value_t>(x_size);
                auto sample_variance = ad::sum(util::counting_iterator<size_t>(0),
                                               util::counting_iterator<size_t>(x_size),
                                               [&](size_t i) {
                                                    return ad::pow<2>(x.to_ad(ad_vars, cache, i) - sample_mean);
                                               }) / ad::constant<dist_value_t>(x_size);

                // Subsubcase 1: sd -> has param
                if constexpr (SDType::has_param) {
                    return (cache[offset_] = ad_sd,
                            ad::if_else(
                            cache[offset_] > ad::constant(0.),
                            (ad::constant(-0.5 * x_size) / ad::pow<2>(cache[offset_]))
                            * ( ad::pow<2>(ad_mean - sample_mean) + sample_variance )
                            - ( ad::constant<dist_value_t>(x_size) * ad::log(cache[offset_]) ),
                            ad::constant(math::neg_inf<dist_value_t>)
                           ) ); 
                }

                // Subsubcase 2: sd -> has no param 
                // don't cache to precompute
                else {
                    return ad::if_else(
                            ad_sd > ad::constant(0.),
                            (ad::constant(-0.5 * x_size) / ad::pow<2>(ad_sd))
                            * ( ad::pow<2>(ad_mean - sample_mean) + sample_variance )
                            - ( ad::constant<dist_value_t>(x_size) * ad::log(ad_sd) ),
                            ad::constant(math::neg_inf<dist_value_t>)
                           ); 
                }

            }
        }

        // Case 3: x -> vector, mean -> vector, sd -> scalar
        else if constexpr (util::is_vec_v<VarType> &&
                           util::is_vec_v<MeanType> &&
                           util::is_scl_v<SDType>)
        {
            assert(x.size() == mean_.size());
            size_t x_size = x.size();
            auto&& ad_sd = sd_.to_ad(ad_vars, cache);

            // Subcase 1: sd -> has param
            if constexpr (SDType::has_param) {
                return (cache[offset_] = ad_sd,
                        ad::if_else(
                        cache[offset_] > ad::constant(0.),
                        (ad::constant(-0.5) / ad::pow<2>(cache[offset_]))
                        * ad::sum(util::counting_iterator<size_t>(0),
                                  util::counting_iterator<size_t>(x_size),
                                  [&](size_t i) { 
                                    return ad::pow<2>(x.to_ad(ad_vars, cache, i) 
                                                    - mean_.to_ad(ad_vars, cache, i)); 
                                  })
                        - (ad::constant<dist_value_t>(x_size) * ad::log(cache[offset_])),
                        ad::constant(math::neg_inf<dist_value_t>)
                       ) ); 
            }

            // Subcase 2: sd -> has no param
            else {
                return ad::if_else(
                        ad_sd > ad::constant(0.),
                        (ad::constant(-0.5) / ad::pow<2>(ad_sd))
                        * ad::sum(util::counting_iterator<size_t>(0),
                                  util::counting_iterator<size_t>(x_size),
                                  [&](size_t i) { 
                                    return ad::pow<2>(x.to_ad(ad_vars, cache, i) 
                                                    - mean_.to_ad(ad_vars, cache, i)); 
                                  })
                        - (ad::constant<dist_value_t>(x_size) * ad::log(ad_sd)),
                        ad::constant(math::neg_inf<dist_value_t>)
                       ); 
            }

        }

        // Case 4: x -> vector, mean -> scalar, sd -> vector
        else if constexpr (util::is_vec_v<VarType> &&
                           util::is_scl_v<MeanType> &&
                           util::is_vec_v<SDType>) {
            assert(x.size() == sd_.size());
            auto&& ad_mean = mean_.to_ad(ad_vars, cache);

            // Helper lambda to generate sum of subexpressions
            // which depend on ith x and sd when sd > 0
            // Only used in subcase 2 and 3
            auto within_range = [&](auto expr_gen) {
                 return ad::sum(util::counting_iterator<>(0),
                                util::counting_iterator<>(sd_.size()),
                                [&](auto i) {
                                auto&& ad_x_i = x.to_ad(ad_vars, cache, i);
                                auto&& ad_sd_i = sd_.to_ad(ad_vars, cache, i);
                                return ad::if_else(
                                     ad_sd_i > ad::constant(0.),
                                     expr_gen(ad_x_i, ad_sd_i),
                                     ad::constant(math::neg_inf<dist_value_t>) 
                                        );
                                });
            };

            // Subcase 1: sd -> has param
            if constexpr (SDType::has_param) {
                static_cast<void>(within_range);

                return (cache[offset_] = ad_mean,
                        ad::sum(util::counting_iterator<>(0),
                                util::counting_iterator<>(x.size()),
                                [&](auto i) {
                                auto&& ad_x = x.to_ad(ad_vars, cache, i);
                                auto&& ad_sd = sd_.to_ad(ad_vars, cache, i);
                                return (cache[offset_+1+i] = ad_sd,
                                        ad::if_else(
                                            cache[offset_+1+i] > ad::constant(0.),
                                            -ad::constant(0.5)*(
                                                ad::pow<2>((ad_x - cache[offset_])/cache[offset_+1+i]) )
                                            -ad::log(cache[offset_+1+i]),
                                            ad::constant(math::neg_inf<dist_value_t>)
                                        ) );
                                }) );
            } // end case 4, subcase 1

            // Subcase 2: x -> has param, sd -> has no param
            else if constexpr (VarType::has_param) {
                auto&& ad_log_sum = within_range(
                        [](const auto&,
                           const auto& ad_sd_i) {
                            return -ad::log(ad_sd_i);
                        }); 

                // ad_sd is -inf iff there exists i s.d. sd_i <= 0
                return (cache[offset_] = ad_mean,
                        ad::if_else(
                            ad_log_sum != ad::constant(math::neg_inf<dist_value_t>),
                            ad::sum(util::counting_iterator<>(0),
                                    util::counting_iterator<>(x.size()),
                                    [&](auto i){
                                    auto&& ad_sd_i = sd_.to_ad(ad_vars, cache, i);
                                    auto&& ad_x_i = x.to_ad(ad_vars, cache, i);
                                    return (ad::constant(-0.5) / ad::pow<2>(ad_sd_i)) *
                                            ad::pow<2>(ad_x_i - cache[offset_]);
                                    }) 
                            + ad_log_sum,
                            ad_log_sum
                        ) );

            } // end case 4, subcase 2

            // Subcase 3: x -> has no param, sd -> has no param
            // HUGE optimization
            else {

                auto&& ad_log_sum = within_range(
                        [](const auto&,
                           const auto& ad_sd_i) {
                            return -ad::log(ad_sd_i);
                        }); 

                auto&& ad_x_sq = within_range(
                        [](const auto& ad_x_i,
                           const auto& ad_sd_i) {
                        return -ad::constant(0.5) * 
                                ad::pow<2>(ad_x_i/ad_sd_i);
                        });

                auto&& ad_x_ln = within_range(
                        [](const auto& ad_x_i,
                           const auto& ad_sd_i) {
                        return ad_x_i/ad::pow<2>(ad_sd_i);
                        });

                auto&& ad_x_const = within_range(
                        [](const auto&,
                           const auto& ad_sd_i) {
                        return ad::constant(-0.5)/ad::pow<2>(ad_sd_i);
                        });

                return (cache[offset_] = ad_mean,
                        ad::if_else(
                            ad_log_sum != ad::constant(math::neg_inf<dist_value_t>),
                            ad_x_sq + cache[offset_] * ad_x_ln +
                            ad::pow<2>(cache[offset_]) * ad_x_const
                            + ad_log_sum,
                            ad_log_sum
                            ) );

            } // end case 4, subcase 3

        } // end case 4

        // Case 5: x -> vec, mean -> vec, sd -> vec
        else if constexpr (util::is_vec_v<VarType> &&
                           util::is_vec_v<MeanType> &&
                           util::is_vec_v<SDType>) {
            assert(x.size() == mean_.size() &&
                   x.size() == sd_.size());

            return ad::sum(util::counting_iterator<>(0),
                           util::counting_iterator<>(x.size()),
                           [&](auto i) {
                           auto&& ad_x_i = x.to_ad(ad_vars, cache, i);
                           auto&& ad_mean_i = mean_.to_ad(ad_vars, cache, i);
                           auto&& ad_sd_i = sd_.to_ad(ad_vars, cache, i);
                           return (cache[offset_+i] = ad_sd_i,
                                   ad::if_else(
                                        cache[offset_+i] > ad::constant(0.),
                                        ad::constant(-0.5) * 
                                        ad::pow<2>((ad_x_i - ad_mean_i)/cache[offset_+i])
                                        - ad::log(cache[offset_+i]),
                                        ad::constant(math::neg_inf<dist_value_t>)
                                   ) );
                           });
        } // end case 5

    }
        
    template <class PVecType
            , class F = util::identity>
    value_t min(const PVecType&, 
                size_t=0,
                F = F()) const 
    { return math::neg_inf<value_t>; }

    
    template <class PVecType
            , class F = util::identity>
    value_t max(const PVecType&, 
                size_t=0,
                F = F()) const 
    { return math::inf<value_t>; }

    // TODO: impl will change when SDType can be matrix.
    index_t set_cache_offset(index_t idx) 
    {
        idx = mean_.set_cache_offset(idx);
        idx = sd_.set_cache_offset(idx);

        // Case 1: mean -> scalar, sd -> scalar
        // Need to cache both mean and sd
        if constexpr (util::is_scl_v<MeanType> &&
                      util::is_scl_v<SDType>) {
            offset_ = idx;
            return idx + 2;
        }

        // Case 2: mean -> vector, sd -> scalar
        // only need to cache sd when it has param
        else if constexpr (util::is_vec_v<MeanType> &&
                           util::is_scl_v<SDType> &&
                           SDType::has_param) {
            offset_ = idx;
            return idx + 1; 
        }

        // Case 3: mean -> scalar, sd -> vector
        // may need to cache both mean and every element of sd
        else if constexpr (util::is_scl_v<MeanType> &&
                           util::is_vec_v<SDType>) {
            offset_ = idx;

            if constexpr (SDType::has_param) {
                return idx + 1 + sd_.size(); 
            } else {
                return idx + 1;
            }
        }

        // Case 4: mean -> vector, sd -> vector
        else if constexpr (util::is_vec_v<MeanType> &&
                           util::is_vec_v<SDType>) {
            offset_ = idx;
            return idx + sd_.size();
        }

        // Otherwise, don't use cache.
        return idx;
    }

private:
    index_t offset_;
    MeanType mean_; 
    SDType sd_;
};

} // namespace expr
} // namespace ppl

#undef PPL_NORMAL_PARAM_SHAPE
