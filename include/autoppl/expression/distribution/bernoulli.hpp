#pragma once
#include <cassert>
#include <autoppl/util/traits/var_expr_traits.hpp>
#include <autoppl/util/traits/dist_expr_traits.hpp>
#include <autoppl/util/functional.hpp>
#include <autoppl/util/iterator/counting_iterator.hpp>
#include <autoppl/expression/distribution/dist_utils.hpp>
#include <autoppl/math/density.hpp>

#define PPL_BERNOULLI_PARAM_SHAPE \
    "Bernoulli distribution probability must either be a scalar or vector. " \

namespace ppl {
namespace expr {
namespace details {

/**
 * Checks whether PType has proper shape.
 * Must be proper shape and cannot be matrix.
 */
template <class PType>
struct bern_valid_param_dim
{
    static constexpr bool value =
        util::is_shape_v<PType> &&
        !util::is_mat_v<PType>;
};

/**
 * Checks if VarType and PType have proper relative shapes.
 * Currently, only allow up to vector shape (no matrix).
 */
template <class VarType
        , class PType>
struct bern_valid_dim
{
    static constexpr bool value =
        util::is_shape_v<VarType> &&
        (
            (util::is_scl_v<VarType> && 
             util::is_scl_v<PType>) ||
            (util::is_vec_v<VarType> && 
             bern_valid_param_dim<PType>::value)        
        );
};

template <class PType>
inline constexpr bool bern_valid_param_dim_v =
    bern_valid_param_dim<PType>::value;

template <class VarType
        , class PType>
inline constexpr bool bern_valid_dim_v =
    bern_valid_dim<VarType, PType>::value;

} // namespace details

/**
 * Bernoulli is a generic expression representing the
 * Bernoulli distribution.
 * Its parameter type PType must satisfy variable expression.
 * It is tagged as a discrete distribution and satisfies
 * distribution expression.
 *
 * If PType is a vector shape, then this distribution
 * is treated as a joint distribution of n independent Bernoulli
 * (scalar) random variables.
 *
 * @tparam PType    probability variable expression type.
 *                  Cannot be a matrix shape.
 */

template <class PType>
struct Bernoulli : util::DistExprBase<Bernoulli<PType>>
{
    static_assert(util::is_var_expr_v<PType>);
    static_assert(details::bern_valid_param_dim_v<PType>,
                  PPL_DIST_SHAPE_MISMATCH
                  PPL_BERNOULLI_PARAM_SHAPE
                  );

    using value_t = util::disc_param_t;
    using param_value_t = typename util::var_expr_traits<PType>::value_t;
    using base_t = util::DistExprBase<Bernoulli<PType>>;
    using index_t = uint32_t;
    using typename base_t::dist_value_t;

    Bernoulli(const PType& p)
        : p_{p} {}

    template <class VarType
            , class PVecType
            , class F = util::identity>
    dist_value_t pdf(const VarType& x,
                     const PVecType& pvalues,
                     F f = F()) const 
    {
        static_assert(util::is_var_v<VarType>);
        static_assert(details::bern_valid_dim_v<VarType, PType>,
                      PPL_DIST_SHAPE_MISMATCH);
        return pdf_indep([&](size_t i) {
                            return math::bernoulli_pdf(
                                    x.value(pvalues, i, f), 
                                    p_.value(pvalues, i, f));
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
        static_assert(details::bern_valid_dim_v<VarType, PType>,
                      PPL_DIST_SHAPE_MISMATCH);
        return pdf_indep([&](size_t i) {
                            return math::bernoulli_log_pdf(
                                    x.value(pvalues, i, f), 
                                    p_.value(pvalues, i, f));
                          }, x.size());
    }

    template <class VarType, class VecADVarType>
    auto ad_log_pdf(const VarType& x,
                    const VecADVarType& ad_vars,
                    const VecADVarType& cache) const
    { 
        // discrete version of log pdf when 0 < p < 1
        auto p_within_range_disc = [&](const auto& x_ad,
                                       const auto& cache_p) {
            return ad::if_else(
                    x_ad == ad::constant(0.),
                    ad::log(ad::constant(1.)-cache_p),
                    ad::if_else(
                        x_ad == ad::constant(1.),
                        ad::log(cache_p),
                        ad::constant(math::neg_inf<param_value_t>)
                        )
                );
        };

        // continuous version of log pdf when 0 < p < 1
        auto p_within_range_cont = [&](const auto& x_ad,
                                       const auto& cache_p) {
            return ad::constant<param_value_t>(x.size()) * (
                        x_ad * ad::log(cache_p) +
                        (ad::constant(1.) - x_ad) * 
                        ad::log(ad::constant(1.) - cache_p)
                    );
        };

        auto scalar_expr_gen = [](const auto& x_ad,
                                  const auto& cache_p,
                                  auto p_within_range) {
            auto&& clip_upper = ad::if_else(
                    cache_p >= ad::constant(1.),
                    ad::if_else(
                        x_ad == ad::constant(1.),
                        ad::constant(0.),
                        ad::constant(math::neg_inf<param_value_t>)
                        ),
                    p_within_range(x_ad, cache_p)
                );

            auto&& clipped_log_pdf = ad::if_else(
                    cache_p <= ad::constant(0.),
                    ad::if_else(
                        x_ad == ad::constant(0.),
                        ad::constant(0.),
                        ad::constant(math::neg_inf<param_value_t>)
                        ),
                    clip_upper
                );

            return clipped_log_pdf;
        };

        // Case 1: x -> scl, p -> scl
        if constexpr (util::is_scl_v<VarType> &&
                      util::is_scl_v<PType>) {
            static_cast<void>(p_within_range_cont);
            return (cache[offset_] = p_.to_ad(ad_vars, cache),
                    scalar_expr_gen(x.to_ad(ad_vars, cache), 
                                    cache[offset_], 
                                    p_within_range_disc));
        }

        // Case 2: x -> vec, p -> scl
        // HUGE optimization especially when x is data,
        // which is the only time this should ever get called anyway.
        else if constexpr (util::is_vec_v<VarType> &&
                           util::is_scl_v<PType>) {
            static_cast<void>(p_within_range_disc);
            auto&& x_mean = ad::sum(util::counting_iterator<>(0),
                                    util::counting_iterator<>(x.size()),
                                    [&](auto i) {
                                        return x.to_ad(ad_vars, cache, i);
                                    }) / ad::constant<param_value_t>(x.size());

            return (cache[offset_] = x_mean,
                    cache[offset_+1] = p_.to_ad(ad_vars, cache),
                    scalar_expr_gen(cache[offset_],
                                    cache[offset_+1],
                                    p_within_range_cont)
                   );
        }

        // Case 3: x -> vec, p -> vec
        else {
            assert(x.size() == p_.size());
            static_cast<void>(p_within_range_cont);
            return ad::sum(util::counting_iterator<>(0),
                           util::counting_iterator<>(x.size()),
                           [&](auto i) {
                               return (cache[offset_+i] = p_.to_ad(ad_vars, cache, i),
                                       scalar_expr_gen(x.to_ad(ad_vars, cache, i),
                                                       cache[offset_+i],
                                                       p_within_range_disc));
                           });
        }
    }

    /**
     * Requires at most 2 cache variables when p is scalar.
     * When variable is vector but PType scalar,
     * we need to cache the sum of the variable elements.
     * Otherwise, we need to cache every element.
     */
    index_t set_cache_offset(index_t idx) 
    {
        idx = p_.set_cache_offset(idx);

        if constexpr (util::is_scl_v<PType>) {
            offset_ = idx;
            return offset_ + 2;
        }
        else if constexpr (util::is_vec_v<PType>) {
            offset_ = idx;
            return offset_ + p_.size();
        }

        return idx;
    }

    template <class PVecType
            , class F = util::identity>
    value_t min(const PVecType&, 
                size_t=0,
                F = F()) const 
    { return 0; }

    template <class PVecType
            , class F = util::identity>
    value_t max(const PVecType&, 
                size_t=0,
                F = F()) const 
    { return 1; }

private:
    index_t offset_;
    PType p_;
};

} // namespace expr
} // namespace ppl

#undef PPL_BERNOULLI_PARAM_SHAPE
