#pragma once
#include <cassert>
#include <random>
#include <fastad_bits/reverse/stat/uniform.hpp>
#include <autoppl/util/traits/traits.hpp>
#include <autoppl/expression/distribution/dist_utils.hpp>
#include <autoppl/math/density.hpp>
#include <autoppl/math/math.hpp>

#define PPL_UNIFORM_PARAM_SHAPE \
    "Uniform parameters min and max must be either scalar or vector. "

namespace ppl {
namespace expr {
namespace dist {
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
private:
    using min_t = MinType;
    using max_t = MaxType;

    static_assert(util::is_var_expr_v<min_t>);
    static_assert(util::is_var_expr_v<max_t>);
    static_assert(details::uniform_valid_param_dim_v<min_t, max_t>,
                  PPL_DIST_SHAPE_MISMATCH
                  PPL_UNIFORM_PARAM_SHAPE
                  );

public:
    using value_t = util::cont_param_t;
    using base_t = util::DistExprBase<Uniform<min_t, max_t>>; 
    using typename base_t::dist_value_t;

    Uniform(const min_t& min, 
            const max_t& max)
        : min_{min}, max_{max} 
    {}

    template <class XType>
    dist_value_t pdf(const XType& x) 
    {
        static_assert(util::is_dist_assignable_v<XType>);
        static_assert(details::uniform_valid_dim_v<XType, min_t, max_t>,
                      PPL_DIST_SHAPE_MISMATCH);
        return math::uniform_pdf(x.get(), min_.eval(), max_.eval());
    }

    template <class XType>
    dist_value_t log_pdf(const XType& x) 
    {
        static_assert(util::is_dist_assignable_v<XType>);
        static_assert(details::uniform_valid_dim_v<XType, min_t, max_t>,
                      PPL_DIST_SHAPE_MISMATCH);
        return math::uniform_log_pdf(x.get(), min_.eval(), max_.eval());
    }

    template <class XType
            , class PtrPackType>
    auto ad_log_pdf(const XType& x,
                    const PtrPackType& pack) const
    {
        return ad::uniform_adj_log_pdf(x.ad(pack),
                                       min_.ad(pack),
                                       max_.ad(pack));
    }

    template <class PtrPackType>
    void bind(const PtrPackType& pack)
    { 
        static_cast<void>(pack);
        if constexpr (min_t::has_param) {
            min_.bind(pack);
        }
        if constexpr (max_t::has_param) {
            max_.bind(pack);
        }
    }

    void activate_refcnt() const 
    { 
        min_.activate_refcnt(); 
        max_.activate_refcnt();
    }

    // Note: assumes that min_ and max_ have already been evaluated!
    template <class XType, class GenType>
    bool prune(XType& x, GenType& gen) const { 
        using x_t = std::decay_t<XType>;
        static_assert(util::is_param_v<x_t>);

        auto m = min_.get();
        auto M = max_.get();
        std::uniform_real_distribution<dist_value_t> dist(0.,1.);

        if constexpr (util::is_scl_v<x_t>) {
            bool needs_prune = (x.get() <= m) || (x.get() >= M); 
            if (needs_prune) {
                x.get() = (M-m) * dist(gen) + m;
            }
            return needs_prune;

        } else if constexpr (util::is_vec_v<x_t>) {
            auto get = [](const auto& v, size_t i=0, size_t j=0) {
                using v_t = std::decay_t<decltype(v)>;
                static_cast<void>(i);
                static_cast<void>(j);
                if constexpr (!ad::util::is_eigen_v<v_t>) {
                    return v;
                } else {
                    return v(i,j);
                }
            };
            auto to_array = [](const auto& v) {
                using v_t = std::decay_t<decltype(v)>;
                if constexpr (!ad::util::is_eigen_v<v_t>) {
                    return v;
                } else {
                    return v.array();
                }
            };
            
            auto xa = x.get().array();
            bool needs_prune = (xa <= to_array(m)).max(xa >= to_array(M)).any();
            if (needs_prune) {
                using vec_t = std::decay_t<decltype(x.get())>;
                x.get() = vec_t::NullaryExpr(x.get().size(), 
                        [&](size_t i) { 
                            return (get(M, i) - get(m, i)) * dist(gen) + get(m, i);
                        });
            }
            return needs_prune;

        } else {
            static_assert(util::is_scl_v<x_t> ||
                          util::is_vec_v<x_t>, 
                          "x must be a scalar or vector shape.");
        }
    }

private:
    min_t min_; 
    max_t max_;
};

} // namespace dist
} // namespace expr

/**
 * Builds a Uniform expression only when the parameters
 * are both valid continuous distribution parameter types.
 * See var_expr.hpp for more information.
 */
template <class MinType, class MaxType
        , class = std::enable_if_t<
            util::is_valid_dist_param_v<MinType> &&
            util::is_valid_dist_param_v<MaxType>
         > >
inline constexpr auto uniform(const MinType& min_expr,
                              const MaxType& max_expr)
{
    using min_t = util::convert_to_param_t<MinType>;
    using max_t = util::convert_to_param_t<MaxType>;

    min_t wrap_min_expr = min_expr;
    max_t wrap_max_expr = max_expr;

    return expr::dist::Uniform(wrap_min_expr, wrap_max_expr);
}

} // namespace ppl

#undef PPL_UNIFORM_PARAM_SHAPE
