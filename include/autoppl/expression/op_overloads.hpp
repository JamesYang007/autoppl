#pragma once
#include <autoppl/expression/variable/binary.hpp>
#include <autoppl/expression/variable/unary.hpp>
#include <autoppl/expression/variable/glue.hpp>
#include <autoppl/expression/variable/op_eq.hpp>
#include <autoppl/expression/model/bar_eq.hpp>
#include <autoppl/expression/model/glue.hpp>
#include <autoppl/expression/program/program.hpp>
#include <autoppl/util/traits/traits.hpp>

namespace ppl {

// operator- (unary)
PPL_UNARY_FUNC(operator-, UnaryMinus)

// operator+,-,*,/ (binary)
template <class LHSType, class RHSType
        , class = std::enable_if_t<
            util::is_not_both_arithmetic_v<LHSType, RHSType> &&
            util::is_valid_op_param_v<LHSType> &&
            util::is_valid_op_param_v<RHSType>
        > >
inline constexpr auto operator+(const LHSType& lhs,
	                            const RHSType& rhs)
{ 
    return expr::var::details::operator_helper<ad::math::Add>(lhs, rhs);
}

template <class LHSType, class RHSType
        , class = std::enable_if_t<
            util::is_not_both_arithmetic_v<LHSType, RHSType> &&
            util::is_valid_op_param_v<LHSType> &&
            util::is_valid_op_param_v<RHSType>
        > >
inline constexpr auto operator-(const LHSType& lhs, 
                                const RHSType& rhs)
{
    return expr::var::details::operator_helper<ad::math::Sub>(lhs, rhs);
}

template <class LHSType, class RHSType
        , class = std::enable_if_t<
            util::is_not_both_arithmetic_v<LHSType, RHSType> &&
            util::is_valid_op_param_v<LHSType> &&
            util::is_valid_op_param_v<RHSType>
        > >
inline constexpr auto operator*(const LHSType& lhs, 
                                const RHSType& rhs)
{
    return expr::var::details::operator_helper<ad::math::Mul>(lhs, rhs);
}

template <class LHSType, class RHSType
        , class = std::enable_if_t<
            util::is_not_both_arithmetic_v<LHSType, RHSType> &&
            util::is_valid_op_param_v<LHSType> &&
            util::is_valid_op_param_v<RHSType>
        > >
inline constexpr auto operator/(const LHSType& lhs, 
                                const RHSType& rhs)
{
    return expr::var::details::operator_helper<ad::math::Div>(lhs, rhs);
}

// operator+=, -=, *=, /=
template <class TParamViewType
        , class VarExprType
        , class = std::enable_if_t<
            util::is_tparam_v<TParamViewType> &&
            util::is_valid_op_param_v<VarExprType>
        > >
constexpr inline auto operator+=(const TParamViewType& tp_view,
                                 const VarExprType& expr)
{
    return expr::var::details::opeq_helper<expr::var::AddEq>(tp_view, expr);
}

template <class TParamViewType
        , class VarExprType
        , class = std::enable_if_t<
            util::is_tparam_v<TParamViewType> &&
            util::is_valid_op_param_v<VarExprType>
        > >
constexpr inline auto operator-=(const TParamViewType& tp_view,
                                 const VarExprType& expr)
{
    return expr::var::details::opeq_helper<expr::var::SubEq>(tp_view, expr);
}

template <class TParamViewType
        , class VarExprType
        , class = std::enable_if_t<
            util::is_tparam_v<TParamViewType> &&
            util::is_valid_op_param_v<VarExprType>
        > >
constexpr inline auto operator*=(const TParamViewType& tp_view,
                                 const VarExprType& expr)
{
    return expr::var::details::opeq_helper<expr::var::MulEq>(tp_view, expr);
}

template <class TParamViewType
        , class VarExprType
        , class = std::enable_if_t<
            util::is_tparam_v<TParamViewType> &&
            util::is_valid_op_param_v<VarExprType>
        > >
constexpr inline auto operator/=(const TParamViewType& tp_view,
                                 const VarExprType& expr)
{
    return expr::var::details::opeq_helper<expr::var::DivEq>(tp_view, expr);
}

// operator,
template <class LHSExprType
        , class RHSExprType
        , class = std::enable_if_t<
            (util::is_var_expr_v<LHSExprType> ||
            util::is_model_expr_v<LHSExprType>) &&
            (util::is_var_expr_v<RHSExprType> ||
            util::is_model_expr_v<RHSExprType>)
        > >
constexpr inline auto operator,(const LHSExprType& lhs,
                                const RHSExprType& rhs)
{
    if constexpr (util::is_var_expr_v<LHSExprType> &&
                  util::is_var_expr_v<RHSExprType>) {
        using lhs_t = util::convert_to_param_t<LHSExprType>;
        using rhs_t = util::convert_to_param_t<RHSExprType>;
        lhs_t wrap_lhs = lhs;
        rhs_t wrap_rhs = rhs;
        return expr::var::GlueNode<lhs_t, rhs_t>(wrap_lhs, wrap_rhs);

    } else if constexpr (util::is_model_expr_v<LHSExprType> &&
                         util::is_model_expr_v<RHSExprType>) {
        return expr::model::GlueNode(lhs.self(), rhs.self());
    } else {

        static_assert(util::is_var_expr_v<LHSExprType> &&
                      util::is_var_expr_v<RHSExprType>,
                      "Both expressions must be either variable expression or model expressions.");
    }
}

/**
 * Builds an BarEqNode to associate var with dist
 * only when var is a Variable and dist is a valid distribution expression.
 * Ex. x |= uniform(0,1)
 */
template <class VarType
        , class DistType
        , class = std::enable_if_t<
            util::is_var_v<VarType> &&
            util::is_dist_assignable_v<VarType> &&
            util::is_dist_expr_v<DistType>
    > >
inline constexpr auto operator|=(const VarType& var,
                                 const DistType& dist) 
{ 
    using view_t = util::convert_to_param_t<VarType>;
    view_t var_view = var;
    return expr::model::BarEqNode(var_view, dist.self()); 
}

template <class TPExpr
        , class ModelExpr
        , class = std::enable_if_t<
            util::is_var_expr_v<TPExpr> &&
            util::is_model_expr_v<ModelExpr>
        > >
constexpr inline auto operator|(const TPExpr& tp,
                                const ModelExpr& model)
{
    using tp_t = TPExpr;
    using model_t = ModelExpr;
    return expr::prog::ProgramNode<std::tuple<tp_t, model_t>>(tp, model);
}

} // namespace ppl
