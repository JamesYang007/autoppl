#pragma once
#include <autoppl/util/traits.hpp>
#include <autoppl/expression/model/eq_node.hpp>
#include <autoppl/expression/model/glue_node.hpp>
#include <autoppl/expression/variable/variable_viewer.hpp>
#include <autoppl/expression/variable/constant.hpp>
#include <autoppl/expression/variable/binop.hpp>
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
 * - T is an arithmetic type => Constant<T>
 * - is_var_expr_v<T> true => T
 * Assumes each condition is non-overlapping.
 */
template <class T, class = void>
struct convert_to_param
{};

template <class T>
struct convert_to_param<T,
    std::enable_if_t<util::is_var_v<std::decay_t<T>> && 
                    !std::is_arithmetic_v<std::decay_t<T>> &&
                    !util::is_var_expr_v<std::decay_t<T>>
                    >>
{
    using type = expr::VariableViewer<std::decay_t<T>>;
};

template <class T>
struct convert_to_param<T, 
    std::enable_if_t<!util::is_var_v<std::decay_t<T>> && 
                    std::is_arithmetic_v<std::decay_t<T>> &&
                    !util::is_var_expr_v<std::decay_t<T>>
                    >>
{
    using type = expr::Constant<std::decay_t<T>>;
};

template <class T>
struct convert_to_param<T, 
    std::enable_if_t<!util::is_var_v<std::decay_t<T>> && 
                    !std::is_arithmetic_v<std::decay_t<T>> &&
                    util::is_var_expr_v<std::decay_t<T>>
                    >>
{
    using type = T;
};

template <class T>
using convert_to_param_t = 
    typename convert_to_param<T>::type;

/*
 * Checks if valid distribution parameter:
 * - can be arithmetic
 * - if not arithmetic, must be variable or variable expression
 * - if variable, cannot be (rvalue reference or const)
 */
template <class T>
inline constexpr bool is_valid_dist_param_v =
    std::is_arithmetic_v<std::decay_t<T>> ||
    (util::is_var_v<std::decay_t<T>> && 
     !std::is_rvalue_reference_v<T> &&
     !std::is_const_v<std::remove_reference_t<T>>) ||
    (util::is_var_expr_v<std::decay_t<T>>)
    ;

/*
 * Checks if the decayed types of T1 and T2 
 * are not both arithmetic types.
 */
template <class T1, class T2>
inline constexpr bool is_not_both_arithmetic_v =
    !(std::is_arithmetic_v<std::decay_t<T1>> &&
      std::is_arithmetic_v<std::decay_t<T2>>)
    ;

} // namespace details

/*
 * Builds a Uniform expression only when the parameters
 * are both valid continuous distribution parameter types.
 * See var_expr.hpp for more information.
 */
template <class MinType, class MaxType
        , class = std::enable_if_t<
            details::is_valid_dist_param_v<MinType> &&
            details::is_valid_dist_param_v<MaxType>
         > >
inline constexpr auto uniform(MinType&& min_expr,
                              MaxType&& max_expr)
{
    using min_t = details::convert_to_param_t<MinType>;
    using max_t = details::convert_to_param_t<MaxType>;

    min_t wrap_min_expr = std::forward<MinType>(min_expr);
    max_t wrap_max_expr = std::forward<MaxType>(max_expr);

    return expr::Uniform(wrap_min_expr, wrap_max_expr);
}

/*
 * Builds a Normal expression only when the parameters
 * are both valid continuous distribution parameter types.
 * See var_expr.hpp for more information.
 */
template <class MeanType, class StddevType
        , class = std::enable_if_t<
            details::is_valid_dist_param_v<MeanType> &&
            details::is_valid_dist_param_v<StddevType>
         > >
inline constexpr auto normal(MeanType&& mean_expr,
                             StddevType&& stddev_expr)
{
    using mean_t = details::convert_to_param_t<MeanType>;
    using stddev_t = details::convert_to_param_t<StddevType>;

    mean_t wrap_mean_expr = std::forward<MeanType>(mean_expr);
    stddev_t wrap_stddev_expr = std::forward<StddevType>(stddev_expr);

    return expr::Normal(wrap_mean_expr, wrap_stddev_expr);
}

/*
 * Builds a Bernoulli expression only when the parameter
 * is a valid discrete distribution parameter type.
 * See var_expr.hpp for more information.
 */
template <class ProbType
        , class = std::enable_if_t<
            details::is_valid_dist_param_v<ProbType>
        > >
inline constexpr auto bernoulli(ProbType&& p_expr)
{
    using p_t = details::convert_to_param_t<ProbType>;
    p_t wrap_p_expr = std::forward<ProbType>(p_expr);
    return expr::Bernoulli(wrap_p_expr);
}

////////////////////////////////////////////////////////
// Model Expression Builder
////////////////////////////////////////////////////////

/*
 * Builds an EqNode to associate var with dist
 * only when var is a Variable and dist is a valid distribution expression.
 * Ex. x |= uniform(0,1)
 */
template <class VarType, class DistType>
inline constexpr auto operator|=(util::Var<VarType>& var,
           const util::DistExpr<DistType>& dist) { return expr::EqNode(var.self(), dist.self()); }

/*
 * Builds a GlueNode to "glue" the left expression with the right
 * only when both parameters are valid model expressions.
 * Ex. (x |= uniform(0,1), y |= uniform(0, 2))
 */
template <class LHSNodeType, class RHSNodeType>
inline constexpr auto operator,(const util::ModelExpr<LHSNodeType>& lhs,
                                const util::ModelExpr<RHSNodeType>& rhs)
{ return expr::GlueNode(lhs.self(), rhs.self()); }

////////////////////////////////////////////////////////
// Variable Expression Builder
////////////////////////////////////////////////////////

namespace details {

/*
 * Concept of valid variable expression parameter
 * for the operator overloads:
 * - can be arithmetic type
 * - if not arithmetic, must be variable expression
 *   or a variable.
 */
template <class T>
inline constexpr bool is_valid_op_param_v =
    std::is_arithmetic_v<std::decay_t<T>> || 
    util::is_var_expr_v<std::decay_t<T>> ||
    util::is_var_v<std::decay_t<T>>
    ;

template <class Op, class LHSType, class RHSType>
inline constexpr auto operator_helper(LHSType&& lhs, RHSType&& rhs)
{
    // note: may be reference types if converted to itself
	using lhs_t = details::convert_to_param_t<LHSType>;
    using rhs_t = details::convert_to_param_t<RHSType>;

   	lhs_t wrap_lhs_expr = std::forward<LHSType>(lhs);
    rhs_t wrap_rhs_expr = std::forward<RHSType>(rhs);
    
    using binary_t = expr::BinaryOpNode<
        Op, std::decay_t<lhs_t>, std::decay_t<rhs_t>
    >;

    // lhs_t and rhs_t are decayed by node_t
	return binary_t(wrap_lhs_expr, wrap_rhs_expr);
}

} // namespace details

/*
 * Operator overloads, which only check for type-safety.
 * SFINAE to ensure concepts are placed.
 */

template <class LHSType, class RHSType
        , class = std::enable_if_t<
            details::is_not_both_arithmetic_v<LHSType, RHSType> &&
            details::is_valid_op_param_v<LHSType> &&
            details::is_valid_op_param_v<RHSType>
        > >
inline constexpr auto operator+(LHSType&& lhs,
	                            RHSType&& rhs)
{ 
    return details::operator_helper<expr::AddOp>(
            std::forward<LHSType>(lhs),
            std::forward<RHSType>(rhs));
}

template <class LHSType, class RHSType
        , class = std::enable_if_t<
            details::is_not_both_arithmetic_v<LHSType, RHSType> &&
            details::is_valid_op_param_v<LHSType> &&
            details::is_valid_op_param_v<RHSType>
        > >
inline constexpr auto operator-(LHSType&& lhs, RHSType&& rhs)
{
    return details::operator_helper<expr::SubOp>(
            std::forward<LHSType>(lhs),
            std::forward<RHSType>(rhs));
}

template <class LHSType, class RHSType
        , class = std::enable_if_t<
            details::is_not_both_arithmetic_v<LHSType, RHSType> &&
            details::is_valid_op_param_v<LHSType> &&
            details::is_valid_op_param_v<RHSType>
        > >
inline constexpr auto operator*(LHSType&& lhs, RHSType&& rhs)
{
    return details::operator_helper<expr::MultOp>(
            std::forward<LHSType>(lhs),
            std::forward<RHSType>(rhs));
}

template <class LHSType, class RHSType
        , class = std::enable_if_t<
            details::is_not_both_arithmetic_v<LHSType, RHSType> &&
            details::is_valid_op_param_v<LHSType> &&
            details::is_valid_op_param_v<RHSType>
        > >
inline constexpr auto operator/(LHSType&& lhs, RHSType&& rhs)
{
    return details::operator_helper<expr::DivOp>(
            std::forward<LHSType>(lhs),
            std::forward<RHSType>(rhs));
}

} // namespace ppl
