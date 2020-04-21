#pragma once
#include <autoppl/util/traits.hpp>
#include <autoppl/expression/model/eq_node.hpp>
#include <autoppl/expression/model/glue_node.hpp>
#include <autoppl/expression/variable/variable_viewer.hpp>
#include <autoppl/expression/variable/constant.hpp>
#include <autoppl/variable.hpp>
#include <autoppl/expression/distribution/uniform.hpp>
#include <autoppl/expression/distribution/normal.hpp>
#include <autoppl/expression/distribution/bernoulli.hpp>

namespace ppl {

/*
 * The purpose for these expression builders is to
 * add extra type-safety and ease the user API.
 */

////////////////////////////////////////////////////////
// Distribution Expression Builder
////////////////////////////////////////////////////////

namespace details {

/*
 * Converter from arbitrary (decayed) type to valid continuous parameter type 
 * by the following mapping:
 * - is_var_v<T> true => VariableViewer<T>
 * - T is same as cont_param_t => Constant<T>
 * - is_var_expr_v<T> true => T
 * Assumes each condition is non-overlapping.
 */
template <class T, class = void>
struct convert_to_cont_dist_param
{};

template <class T>
struct convert_to_cont_dist_param<T,
    std::enable_if_t<util::is_var_v<std::decay_t<T>> && 
                    !std::is_same_v<std::decay_t<T>, util::cont_param_t> &&
                    !util::is_var_expr_v<std::decay_t<T>>
                    >>
{
    using type = expr::VariableViewer<std::decay_t<T>>;
};

template <class T>
struct convert_to_cont_dist_param<T, 
    std::enable_if_t<!util::is_var_v<std::decay_t<T>> && 
                    std::is_same_v<std::decay_t<T>, util::cont_param_t> &&
                    !util::is_var_expr_v<std::decay_t<T>>
                    >>
{
    using type = expr::Constant<std::decay_t<T>>;
};

template <class T>
struct convert_to_cont_dist_param<T, 
    std::enable_if_t<!util::is_var_v<std::decay_t<T>> && 
                    !std::is_same_v<std::decay_t<T>, util::cont_param_t> &&
                    util::is_var_expr_v<std::decay_t<T>>
                    >>
{
    using type = T;
};

template <class T>
using convert_to_cont_dist_param_t = 
    typename convert_to_cont_dist_param<T>::type;

} // namespace details

#ifndef AUTOPPL_USE_CONCEPTS
/*
 * Builds a Uniform expression only when the parameters
 * are both valid continuous distribution parameter types.
 * See var_expr.hpp for more information.
 */
template <class MinType, class MaxType>
inline constexpr auto uniform(MinType&& min_expr,
                              MaxType&& max_expr)
{
    using min_t = details::convert_to_cont_dist_param_t<MinType>;
    using max_t = details::convert_to_cont_dist_param_t<MaxType>;

    min_t wrap_min_expr = std::forward<MinType>(min_expr);
    max_t wrap_max_expr = std::forward<MaxType>(max_expr);

    return expr::Uniform(wrap_min_expr, wrap_max_expr);
}
#else
#endif

#ifndef AUTOPPL_USE_CONCEPTS
/*
 * Builds a Normal expression only when the parameters
 * are both valid continuous distribution parameter types.
 * See var_expr.hpp for more information.
 */
template <class MeanType, class StddevType>
inline constexpr auto normal(MeanType&& mean_expr,
                             StddevType&& stddev_expr)
{
    using mean_t = details::convert_to_cont_dist_param_t<MeanType>;
    using stddev_t = details::convert_to_cont_dist_param_t<StddevType>;

    mean_t wrap_mean_expr = std::forward<MeanType>(mean_expr);
    stddev_t wrap_stddev_expr = std::forward<StddevType>(stddev_expr);

    return expr::Normal(wrap_mean_expr, wrap_stddev_expr);
}

#else
#endif

#ifndef AUTOPPL_USE_CONCEPTS
/*
 * Builds a Bernoulli expression only when the parameter
 * is a valid discrete distribution parameter type.
 * See var_expr.hpp for more information.
 * TODO: generalize as done with uniform and normal
 */
template <class ProbType>
inline constexpr auto bernoulli(const ProbType& p_expr)
{
    return expr::Bernoulli(p_expr);
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
inline constexpr auto operator|=(Variable<T>& var,
                                 DistType&& dist)
{ return expr::EqNode(var, std::forward<DistType>(dist)); }
#else
#endif

#ifndef AUTOPPL_USE_CONCEPTS
/*
 * Builds a GlueNode to "glue" the left expression with the right
 * only when both parameters are valid model expressions.
 * Ex. (x |= uniform(0,1), y |= uniform(0, 2))
 */
template <class LHSNodeType, class RHSNodeType>
inline constexpr auto operator,(LHSNodeType&& lhs,
                                RHSNodeType&& rhs)
{ 
    return expr::GlueNode(std::forward<LHSNodeType>(lhs), 
                          std::forward<RHSNodeType>(rhs)); 
}
#else
#endif

} // namespace ppl
