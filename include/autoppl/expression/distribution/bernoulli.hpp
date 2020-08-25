#pragma once
#include <cassert>
#include <fastad_bits/reverse/stat/bernoulli.hpp>
#include <autoppl/util/traits/traits.hpp>
#include <autoppl/expression/distribution/dist_utils.hpp>
#include <autoppl/math/density.hpp>

#define PPL_BERNOULLI_PARAM_SHAPE \
    "Bernoulli distribution probability must either be a scalar or vector. " \

namespace ppl {
namespace expr {
namespace dist {
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
 * Bernoulli is a generic expression representing the Bernoulli distribution.
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
private:
    using p_t = PType;

    static_assert(util::is_var_expr_v<p_t>);
    static_assert(details::bern_valid_param_dim_v<p_t>,
                  PPL_DIST_SHAPE_MISMATCH
                  PPL_BERNOULLI_PARAM_SHAPE
                  );

public:
    using value_t = util::disc_param_t;
    using param_value_t = typename util::var_expr_traits<p_t>::value_t;
    using base_t = util::DistExprBase<Bernoulli<p_t>>;
    using typename base_t::dist_value_t;

    Bernoulli(const p_t& p)
        : p_{p} {}

    template <class XType>
    dist_value_t pdf(const XType& x) 
    {
        static_assert(util::is_dist_assignable_v<XType>);
        static_assert(details::bern_valid_dim_v<XType, p_t>,
                      PPL_DIST_SHAPE_MISMATCH);
        return math::bernoulli_pdf(x.get(), p_.eval());
    }

    template <class XType>
    dist_value_t log_pdf(const XType& x) 
    {
        static_assert(util::is_dist_assignable_v<XType>);
        static_assert(details::bern_valid_dim_v<XType, p_t>,
                      PPL_DIST_SHAPE_MISMATCH);
        return math::bernoulli_log_pdf(x.get(), p_.eval());
    }

    template <class XType
            , class PtrPackType>
    auto ad_log_pdf(const XType& x,
                    const PtrPackType& pack) const
    { 
        return ad::bernoulli_adj_log_pdf(x.ad(pack),
                                         p_.ad(pack));
    }

    template <class PtrPackType>
    void bind(const PtrPackType& pack)
    { 
        static_cast<void>(pack);
        if constexpr (p_t::has_param) {
            p_.bind(pack);
        }
    }

    void activate_refcnt() const 
    { p_.activate_refcnt(); }

    template <class XType, class GenType>
    bool prune(XType& x, GenType&) const {
        using x_t = std::decay_t<XType>;
        static_assert(util::is_param_v<x_t>);
        if constexpr (util::is_scl_v<x_t>) {
            bool needs_prune = (x.get() != 0) && (x.get() != 1); 
            if (needs_prune) x.get() = 0;
            return needs_prune; 
        } else if constexpr (util::is_vec_v<x_t>){
            auto xa = x.get().array();
            bool needs_prune = ((xa != 0).min(xa != 1)).any();
            if (needs_prune) x.get().setZero();
            return needs_prune;
        }
    }

private:
    p_t p_;
};

} // namespace dist
} // namespace expr

/**
 * Builds a Bernoulli expression only when the parameter
 * is a valid discrete distribution parameter type.
 * See var_expr.hpp for more information.
 */
template <class ProbType
        , class = std::enable_if_t<
            util::is_valid_dist_param_v<ProbType>
        > >
inline constexpr auto bernoulli(const ProbType& p_expr)
{
    using p_t = util::convert_to_param_t<ProbType>;
    p_t wrap_p_expr = p_expr;
    return expr::dist::Bernoulli(wrap_p_expr);
}

} // namespace ppl

#undef PPL_BERNOULLI_PARAM_SHAPE
