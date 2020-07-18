#pragma once
#include <autoppl/util/traits/traits.hpp>
#include <autoppl/expression/model/eq_node.hpp>
#include <autoppl/expression/model/glue_node.hpp>
#include <autoppl/expression/variable/constant.hpp>
#include <autoppl/expression/variable/data.hpp>
#include <autoppl/expression/variable/param.hpp>
#include <autoppl/expression/variable/binop.hpp>
#include <autoppl/expression/variable/dot.hpp>
#include <autoppl/expression/distribution/uniform.hpp>
#include <autoppl/expression/distribution/normal.hpp>
#include <autoppl/expression/distribution/bernoulli.hpp>

namespace ppl {

/**
 * The purpose for these expression builders is to
 * add extra type-safety and ease the user API.
 */

////////////////////////////////////////////////////////
// Distribution Expression Builder
////////////////////////////////////////////////////////

namespace details {

/**
 * Converter from arbitrary (decayed) type to valid continuous parameter type 
 * by the following mapping:
 * - is_var_v<T> true => convert to corresponding viewer
 * - else if T is an arithmetic type => Constant<T>
 * - else if is_var_expr_v<T> true => T
 * Assumes each condition is non-overlapping.
 */

template <class T, class = void>
struct convert_to_param
{};

// Convert from param to param viewer
template <class T>
struct convert_to_param<T,
    std::enable_if_t<
        util::is_param_v<std::decay_t<T>> &&
        util::is_scl_v<std::decay_t<T>>
    > >
{
private:
    using raw_t = std::decay_t<T>;
    using pointer_t = typename 
        util::param_traits<raw_t>::pointer_t;
public:
    using type = ppl::ParamView<pointer_t, ppl::scl>;
};

template <class T>
struct convert_to_param<T,
    std::enable_if_t<
        util::is_param_v<std::decay_t<T>> &&
        util::is_vec_v<std::decay_t<T>>
    > >
{
private:
    using raw_t = std::decay_t<T>;
    using vec_t = typename 
        util::param_traits<raw_t>::vec_t;
public:
    using type = ppl::ParamView<vec_t, ppl::vec>;
};

// Convert from data to data viewer
template <class T>
struct convert_to_param<T,
    std::enable_if_t<
        util::is_data_v<std::decay_t<T>> &&
        util::is_scl_v<std::decay_t<T>>
    > >
{
private:
    using raw_t = std::decay_t<T>;
    using value_t = typename 
        util::data_traits<raw_t>::value_t;
public:
    using type = ppl::DataView<value_t, ppl::scl>;
};

template <class T>
struct convert_to_param<T,
    std::enable_if_t<
        util::is_data_v<std::decay_t<T>> &&
        util::is_vec_v<std::decay_t<T>>
    > >
{
private:
    using raw_t = std::decay_t<T>;
    using vec_t = typename 
        util::data_traits<raw_t>::vec_t;
public:
    using type = ppl::DataView<vec_t, ppl::vec>;
};

template <class T>
struct convert_to_param<T,
    std::enable_if_t<
        util::is_data_v<std::decay_t<T>> &&
        util::is_mat_v<std::decay_t<T>>
    > >
{
private:
    using raw_t = std::decay_t<T>;
    using mat_t = typename 
        util::data_traits<raw_t>::mat_t;
public:
    using type = ppl::DataView<mat_t, ppl::mat>;
};

// Convert arithmetic types to Constants
template <class T>
struct convert_to_param<T, 
    std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>> >
    >
{
    using type = expr::Constant<std::decay_t<T>>;
};

// Convert variable expressions (not variables) into itself (no change)
template <class T>
struct convert_to_param<T, 
    std::enable_if_t<
        util::is_var_expr_v<std::decay_t<T>> &&
        !util::is_var_v<std::decay_t<T>> > 
    >
{
    using type = T;
};

template <class T>
using convert_to_param_t = 
    typename convert_to_param<T>::type;

/**
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

/**
 * Checks if the decayed types of T1 and T2 
 * are not both arithmetic types.
 */
template <class T1, class T2>
inline constexpr bool is_not_both_arithmetic_v =
    !(std::is_arithmetic_v<std::decay_t<T1>> &&
      std::is_arithmetic_v<std::decay_t<T2>>)
    ;

} // namespace details

/**
 * Builds a Uniform expression only when the parameters
 * are both valid continuous distribution parameter types.
 * See var_expr.hpp for more information.
 */
template <class MinType, class MaxType
        , class = std::enable_if_t<
            details::is_valid_dist_param_v<MinType> &&
            details::is_valid_dist_param_v<MaxType>
         > >
inline constexpr auto uniform(const MinType& min_expr,
                              const MaxType& max_expr)
{
    using min_t = details::convert_to_param_t<MinType>;
    using max_t = details::convert_to_param_t<MaxType>;

    min_t wrap_min_expr = min_expr;
    max_t wrap_max_expr = max_expr;

    return expr::Uniform(wrap_min_expr, wrap_max_expr);
}

/**
 * Builds a Normal expression only when the parameters
 * are both valid continuous distribution parameter types.
 * See var_expr.hpp for more information.
 */
template <class MeanType, class SDType
        , class = std::enable_if_t<
            details::is_valid_dist_param_v<MeanType> &&
            details::is_valid_dist_param_v<SDType>
         > >
inline constexpr auto normal(const MeanType& mean_expr,
                             const SDType& sd_expr)
{
    using mean_t = details::convert_to_param_t<MeanType>;
    using sd_t = details::convert_to_param_t<SDType>;

    mean_t wrap_mean_expr = mean_expr;
    sd_t wrap_sd_expr = sd_expr;

    return expr::Normal(wrap_mean_expr, wrap_sd_expr);
}

/**
 * Builds a Bernoulli expression only when the parameter
 * is a valid discrete distribution parameter type.
 * See var_expr.hpp for more information.
 */
template <class ProbType
        , class = std::enable_if_t<
            details::is_valid_dist_param_v<ProbType>
        > >
inline constexpr auto bernoulli(const ProbType& p_expr)
{
    using p_t = details::convert_to_param_t<ProbType>;
    p_t wrap_p_expr = p_expr;
    return expr::Bernoulli(wrap_p_expr);
}

////////////////////////////////////////////////////////
// Model Expression Builder
////////////////////////////////////////////////////////

/**
 * Builds an EqNode to associate var with dist
 * only when var is a Variable and dist is a valid distribution expression.
 * Ex. x |= uniform(0,1)
 */
template <class VarType
        , class DistType
        , class = std::enable_if_t<
            util::is_var_v<VarType> &&
            util::is_dist_expr_v<DistType>
    > >
inline constexpr auto operator|=(const VarType& var,
                                 const DistType& dist) 
{ 
    using view_t = details::convert_to_param_t<VarType>;
    view_t var_view = var;
    return expr::EqNode(var_view, dist); 
}

/**
 * Builds a GlueNode to "glue" the left expression with the right
 * only when both parameters are valid model expressions.
 * Ex. (x |= uniform(0,1), y |= uniform(0,2))
 */
template <class LHSNodeType
        , class RHSNodeType
        , class = std::enable_if_t<
            util::is_model_expr_v<LHSNodeType> &&
            util::is_model_expr_v<RHSNodeType>
    > >
inline constexpr auto operator,(const LHSNodeType& lhs,
                                const RHSNodeType& rhs)
{ return expr::GlueNode(lhs.self(), rhs.self()); }

////////////////////////////////////////////////////////
// Variable Expression Builder
////////////////////////////////////////////////////////

namespace details {

/**
 * Concept of valid variable expression parameter
 * for the operator overloads:
 * - can be arithmetic type
 * - if not arithmetic, must be variable expression
 *   or a variable.
 */
template <class T>
inline constexpr bool is_valid_op_param_v =
    std::is_arithmetic_v<std::decay_t<T>> || 
    util::is_var_expr_v<std::decay_t<T>>
    ;

template <class Op, class LHSType, class RHSType>
inline constexpr auto operator_helper(const LHSType& lhs, 
                                      const RHSType& rhs)
{
    // note: may be reference types if converted to itself
	using lhs_t = details::convert_to_param_t<LHSType>;
    using rhs_t = details::convert_to_param_t<RHSType>;

   	lhs_t wrap_lhs_expr = lhs;
    rhs_t wrap_rhs_expr = rhs;
    
    using binary_t = expr::BinaryOpNode<Op, lhs_t, rhs_t>;

	return binary_t(wrap_lhs_expr, wrap_rhs_expr);
}

} // namespace details

/**
 * Operator overloads, which only check for type-safety.
 * SFINAE to ensure concepts are placed.
 */

template <class LHSType, class RHSType
        , class = std::enable_if_t<
            details::is_not_both_arithmetic_v<LHSType, RHSType> &&
            details::is_valid_op_param_v<LHSType> &&
            details::is_valid_op_param_v<RHSType>
        > >
inline constexpr auto operator+(const LHSType& lhs,
	                            const RHSType& rhs)
{ 
    return details::operator_helper<expr::AddOp>(lhs, rhs);
}

template <class LHSType, class RHSType
        , class = std::enable_if_t<
            details::is_not_both_arithmetic_v<LHSType, RHSType> &&
            details::is_valid_op_param_v<LHSType> &&
            details::is_valid_op_param_v<RHSType>
        > >
inline constexpr auto operator-(const LHSType& lhs, 
                                const RHSType& rhs)
{
    return details::operator_helper<expr::SubOp>(lhs, rhs);
}

template <class LHSType, class RHSType
        , class = std::enable_if_t<
            details::is_not_both_arithmetic_v<LHSType, RHSType> &&
            details::is_valid_op_param_v<LHSType> &&
            details::is_valid_op_param_v<RHSType>
        > >
inline constexpr auto operator*(const LHSType& lhs, 
                                const RHSType& rhs)
{
    return details::operator_helper<expr::MultOp>(lhs, rhs);
}

template <class LHSType, class RHSType
        , class = std::enable_if_t<
            details::is_not_both_arithmetic_v<LHSType, RHSType> &&
            details::is_valid_op_param_v<LHSType> &&
            details::is_valid_op_param_v<RHSType>
        > >
inline constexpr auto operator/(const LHSType& lhs, 
                                const RHSType& rhs)
{
    return details::operator_helper<expr::DivOp>(lhs, rhs);
}

/**
 * Builds a dot product expression for two expressions.
 */
template <class LHSVarExprType
        , class RHSVarExprType
        , class = std::enable_if_t<
            util::is_var_expr_v<LHSVarExprType> &&
            util::is_var_expr_v<RHSVarExprType>
        > >
inline constexpr auto dot(const LHSVarExprType& lhs,
                          const RHSVarExprType& rhs)
{
	using lhs_t = details::convert_to_param_t<LHSVarExprType>;
    using rhs_t = details::convert_to_param_t<RHSVarExprType>;

   	lhs_t wrap_lhs_expr = lhs;
    rhs_t wrap_rhs_expr = rhs;
    
    return expr::DotNode<lhs_t, rhs_t>(
            wrap_lhs_expr, wrap_rhs_expr);
}

} // namespace ppl
