#pragma once
#include <cassert>
#include <fastad_bits/node.hpp>
#include <fastad_bits/math.hpp>
#include <fastad_bits/ifelse.hpp>
#include <autoppl/util/traits/dist_expr_traits.hpp>
#include <autoppl/util/traits/var_expr_traits.hpp>
#include <autoppl/util/iterator/counting_iterator.hpp>
#include <autoppl/util/functional.hpp>
#include <autoppl/math/density.hpp>
#include <autoppl/math/math.hpp>
#include <autoppl/expression/distribution/dist_utils.hpp>

#define PPL_UNIFORM_PARAM_SHAPE \
    "Uniform parameters min and max must be either scalar or vector. "

namespace ppl {
namespace expr {
namespace details {

/**
 * Checks whether min, max have proper relative shapes.
 * Must be proper shapes and cannot be matrices.
 */
template <class MinType
        , class MaxType>
struct uniform_valid_param_dim
{
    static constexpr bool value =
        util::is_shape_v<MinType> &&
        util::is_shape_v<MaxType> &&
        !util::is_mat_v<MinType> &&
        !util::is_mat_v<MaxType>;
};

/**
 * Checks if var, min, max have proper relative shapes.
 * Currently, we only allow up to vector dimension (no matrix).
 */
template <class VarType
        , class MinType
        , class MaxType>
struct uniform_valid_dim
{
    static constexpr bool value =
        util::is_shape_v<VarType> &&
        (
            (util::is_scl_v<VarType> && 
             util::is_scl_v<MinType> &&
             util::is_scl_v<MaxType>) ||
            (util::is_vec_v<VarType> && 
             uniform_valid_param_dim<MinType, MaxType>::value)        
        );
};

template <class MinType
        , class MaxType>
inline constexpr bool uniform_valid_param_dim_v =
    uniform_valid_param_dim<MinType, MaxType>::value;

template <class VarType
        , class MinType
        , class MaxType>
inline constexpr bool uniform_valid_dim_v =
    uniform_valid_dim<VarType, MinType, MaxType>::value;

} // namespace details

/**
 * Uniform is a generic expression type for the uniform distribution.
 *
 * If MinType or MaxType is a vector, then the variable assigned
 * to this distribution must also be a vector.
 *
 * @tparam  MinType     variable expression for the min.
 *                      Cannot be a matrix shape.
 * @tparam  MaxType     variable expression for the max.
 *                      Cannot be a matrix shape.
 */

template <class MinType
        , class MaxType>
struct Uniform: util::DistExprBase<Uniform<MinType, MaxType>>
{
    static_assert(util::is_var_expr_v<MinType>);
    static_assert(util::is_var_expr_v<MaxType>);
    static_assert(details::uniform_valid_param_dim_v<MinType, MaxType>,
                  PPL_DIST_SHAPE_MISMATCH
                  PPL_UNIFORM_PARAM_SHAPE
                  );

    using value_t = util::cont_param_t;
    using base_t = util::DistExprBase<Uniform<MinType, MaxType>>; 
    using index_t = uint32_t;
    using typename base_t::dist_value_t;

    Uniform(const MinType& min, 
            const MaxType& max)
        : min_{min}, max_{max} 
    {}

    template <class VarType
            , class PVecType
            , class F = util::identity>
    dist_value_t pdf(const VarType& x,
                     const PVecType& pvalues,
                     F f = F()) const
    {
        static_assert(util::is_var_v<VarType>);
        static_assert(details::uniform_valid_dim_v<VarType, MinType, MaxType>,
                      PPL_DIST_SHAPE_MISMATCH);
        return pdf_indep([&](size_t i) {
                            return math::uniform_pdf(
                                    x.value(pvalues, i, f), 
                                    min_.value(pvalues, i, f), 
                                    max_.value(pvalues, i, f));
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
        static_assert(details::uniform_valid_dim_v<VarType, MinType, MaxType>,
                      PPL_DIST_SHAPE_MISMATCH);
        return log_pdf_indep([&](size_t i) {
                                return math::uniform_log_pdf(
                                        x.value(pvalues, i, f), 
                                        min_.value(pvalues, i, f), 
                                        max_.value(pvalues, i, f));
                              }, x.size());
    }

    /**
     * Up to constant addition, returns ad expression of log pdf
     */
    template <class VarType, class VecADVarType>
    auto ad_log_pdf(const VarType& x,
                    const VecADVarType& vars,
                    const VecADVarType& cache) const
    {

        // Case 1: x -> scl, min -> scl, max -> scl
        if constexpr (util::is_scl_v<VarType> &&
                      util::is_scl_v<MinType> &&
                      util::is_scl_v<MaxType>) {
            auto&& ad_x = x.to_ad(vars, cache);
            auto&& ad_min = min_.to_ad(vars, cache);
            auto&& ad_max = max_.to_ad(vars, cache);

            return (cache[offset_] = ad_min,
                    cache[offset_+1] = ad_max,
                    ad::if_else(
                        (cache[offset_] < ad_x) && (ad_x < cache[offset_+1]),
                         -ad::log(cache[offset_+1] - cache[offset_]),
                         ad::constant(math::neg_inf<dist_value_t>)
                    ));
        }

        // Case 2: x -> vec, min -> scl, max -> scl
        else if constexpr (util::is_vec_v<VarType> &&
                           util::is_scl_v<MinType> &&
                           util::is_scl_v<MaxType>) 
        {
            auto&& ad_min = min_.to_ad(vars, cache);
            auto&& ad_max = max_.to_ad(vars, cache);

            // Subcase 1: x -> has no param
            if constexpr (!VarType::has_param) {

                // Note: value can be used instead of to_ad because
                // vars will be ignored by anything that does not have param
                // and here we guaranteed that x has no params.
                
                auto x_min = math::min(util::counting_iterator<>(0),
                                       util::counting_iterator<>(x.size()),
                                       [&](auto i) { return x.value(vars, i); });
                auto x_max = math::max(util::counting_iterator<>(0),
                                       util::counting_iterator<>(x.size()),
                                       [&](auto i) { return x.value(vars, i); });
                return (cache[offset_] = ad_min,
                        cache[offset_+1] = ad_max,
                        ad::if_else(
                            ((cache[offset_] < ad::constant(x_min)) && 
                             (ad::constant(x_max) < cache[offset_+1])),
                            -ad::constant<dist_value_t>(x.size()) *
                            ad::log(cache[offset_+1] - cache[offset_]),
                            ad::constant(math::neg_inf<dist_value_t>)
                        ) );
            }

            // Subcase 2: x -> has param
            else {
                return (cache[offset_] = ad_min,
                        cache[offset_+1] = ad_max,
                        -ad::constant<dist_value_t>(x.size()) *
                        ad::log(cache[offset_+1] - cache[offset_])) 
                        + ad::sum(util::counting_iterator<>(0),
                                  util::counting_iterator<>(x.size()),
                                  [&](auto i) {
                                    return ad::if_else(
                                        ( (cache[offset_] < x.to_ad(vars, cache, i)) && 
                                          (x.to_ad(vars, cache, i) < cache[offset_+1]) ),
                                        ad::constant<dist_value_t>(0),
                                        ad::constant(math::neg_inf<dist_value_t>)
                                        );
                                  }
                            );
            }
        }

        // Case 3: x -> vec, min -> vec, max -> scl
        else if constexpr (util::is_vec_v<VarType> &&
                           util::is_vec_v<MinType> &&
                           util::is_scl_v<MaxType>) {

            assert(x.size() == min_.size());
            auto&& ad_max = max_.to_ad(vars, cache);
            return (cache[offset_] = ad_max,
                    ad::sum(util::counting_iterator<>(0),
                            util::counting_iterator<>(x.size()),
                            [&](auto i) {
                                 auto&& ad_x = x.to_ad(vars, cache, i);
                                 auto&& ad_min = min_.to_ad(vars, cache, i);
                                 return (cache[offset_+1+i] = ad_min,
                                         ad::if_else(
                                             (cache[offset_+1+i] < ad_x) && (ad_x < cache[offset_]),
                                             -ad::log(cache[offset_] - cache[offset_+1+i]),
                                             ad::constant(math::neg_inf<dist_value_t>)
                                        ) );
                            }) 
                    );
        }

        // Case 4: x -> vec, min -> scl, max -> vec
        else if constexpr (util::is_vec_v<VarType> &&
                           util::is_scl_v<MinType> &&
                           util::is_vec_v<MaxType>) {

            assert(x.size() == max_.size());
            auto&& ad_min = min_.to_ad(vars, cache);
            return (cache[offset_] = ad_min,
                    ad::sum(util::counting_iterator<>(0),
                            util::counting_iterator<>(x.size()),
                            [&](auto i) {
                                 auto&& ad_x = x.to_ad(vars, cache, i);
                                 auto&& ad_max = max_.to_ad(vars, cache, i);
                                 return (cache[offset_+1+i] = ad_max,
                                         ad::if_else(
                                             (cache[offset_] < ad_x) && (ad_x < cache[offset_+1+i]),
                                             -ad::log(cache[offset_+1+i] - cache[offset_]),
                                             ad::constant(math::neg_inf<dist_value_t>)
                                        ) );
                            }) 
                    );
        }

        // Case 5: x -> vec, min -> vec, max -> vec
        else {

            assert(x.size() == max_.size() && 
                    x.size() == min_.size());

            return ad::sum(util::counting_iterator<>(0),
                           util::counting_iterator<>(x.size()),
                           [&](auto i) {
                                auto&& ad_x = x.to_ad(vars, cache, i);
                                auto&& ad_min = min_.to_ad(vars, cache, i);
                                auto&& ad_max = max_.to_ad(vars, cache, i);
                                return (cache[offset_+i] = ad_min,
                                        cache[offset_+i+1] = ad_max,
                                        ad::if_else(
                                            (cache[offset_+i] < ad_x) && (ad_x < cache[offset_+i+1]),
                                            -ad::log(cache[offset_+i+1] - cache[offset_+i]),
                                            ad::constant(math::neg_inf<dist_value_t>)
                                       ) );
                           });
        }

    }

    template <class PVecType
            , class F = util::identity>
    value_t min(const PVecType& pvalues, 
                size_t i=0,
                F f = F()) const 
    { return min_.value(pvalues, i, f); }

    template <class PVecType
            , class F = util::identity>
    value_t max(const PVecType& pvalues, 
                size_t i=0,
                F f = F()) const 
    { return max_.value(pvalues, i, f); }

    index_t set_cache_offset(index_t idx) 
    { 
        idx = min_.set_cache_offset(idx);
        idx = max_.set_cache_offset(idx);
        offset_ = idx;
        return idx + min_.size() + max_.size(); 
    }

private:
    index_t offset_;
    MinType min_; 
    MaxType max_;
};

} // namespace expr
} // namespace ppl

#undef PPL_UNIFORM_PARAM_SHAPE
