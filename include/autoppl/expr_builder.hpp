#pragma once
#include <autoppl/expression/model_expr.hpp>
#include <autoppl/expression/var_expr.hpp>
#include <autoppl/expression/model.hpp>
#include <autoppl/expression/variable.hpp>
#include <autoppl/distribution/dist_expr.hpp>
#include <autoppl/distribution/uniform.hpp>
#include <autoppl/distribution/normal.hpp>
#include <autoppl/distribution/bernoulli.hpp>

namespace ppl {

/*
 * The purpose for these expression builders is to
 * add extra type-safety and ease the user API.
 */

////////////////////////////////////////////////////////
// Distribution Expression Builder
////////////////////////////////////////////////////////

namespace details {

using cont_raw_param_t = double;
template <class T>
inline constexpr bool is_cont_param_valid =
    expr::is_var_expr_v<T> || 
    std::is_convertible_v<T, cont_raw_param_t>;

} // namespace details

#ifndef AUTOPPL_USE_CONCEPTS
/*
 * Builds a Uniform expression only when the parameters
 * are both valid continuous distribution parameter types.
 * See var_expr.hpp for more information.
 */
template <class MinType, class MaxType>
inline constexpr auto uniform(const MinType& min_expr,
                              const MaxType& max_expr)
{
    static_assert(details::is_cont_param_valid<MinType>);
    static_assert(details::is_cont_param_valid<MaxType>);
    return Uniform(min_expr, max_expr);
}
#else
#endif

#ifndef AUTOPPL_USE_CONCEPTS
/*
 * Builds a Normal expression only when the parameters
 * are both valid continuous distribution parameter types.
 * See var_expr.hpp for more information.
 */
template <class MeanType, class VarianceType>
inline constexpr auto normal(const MeanType& min_expr,
                             const VarianceType& max_expr)
{
    static_assert(details::is_cont_param_valid<MeanType>);
    static_assert(details::is_cont_param_valid<VarianceType>);
    return Uniform(min_expr, max_expr);
}
#else
#endif

#ifndef AUTOPPL_USE_CONCEPTS
/*
 * Builds a Bernoulli expression only when the parameter
 * is a valid discrete distribution parameter type.
 * See var_expr.hpp for more information.
 */
template <class ProbType>
inline constexpr auto bernoulli(const ProbType& p_expr)
{
    static_assert(details::is_cont_param_valid<ProbType>);
    return Bernoulli(p_expr);
}
#else
#endif

////////////////////////////////////////////////////////
// Model Expression Builder
////////////////////////////////////////////////////////

#ifndef AUTOPPL_USE_CONCEPTS
/*
 * Builds an EqNode to associate var with dist
 * only when var is a Variable and dist is a valid distribution expression.
 * Ex. x |= uniform(0,1)
 */
template <class T, class DistType>
inline constexpr auto operator|=(const Variable<T>& var,
                                 const DistType& dist)
{
    static_assert(dist::is_dist_expr_v<DistType>);
    return EqNode(var, dist);
}
#else
#endif

#ifndef AUTOPPL_USE_CONCEPTS
/*
 * Builds a GlueNode to "glue" the left expression with the right
 * only when both parameters are valid model expressions.
 * Ex. (x |= uniform(0,1), y |= uniform(0, 2))
 */
template <class LHSNodeType, class RHSNodeType>
inline constexpr auto operator,(const LHSNodeType& lhs,
                                const RHSNodeType& rhs)
{
    static_assert(expr::is_model_expr_v<LHSNodeType>);
    static_assert(expr::is_model_expr_v<RHSNodeType>);
    return GlueNode(lhs, rhs);
}
#else
#endif

} // namespace ppl
