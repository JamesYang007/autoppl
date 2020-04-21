#pragma once

#include <type_traits>
#include <autoppl/util/traits.hpp>
#include <autoppl/expression/var_expr.hpp>
#include <autoppl/expression/variable.hpp>

namespace ppl {

template <class LHSValueType, class RHSValueType, class BinaryOp>
struct BinaryOpNode 
{
	using binop_result_t = std::common_type_t<
		typename node_traits<LHSValueType>::binop_result_t,
		typename node_traits<RHSValueType>::binop_result_t
			>;

	BinaryOpNode(Variable<LHSValueType> lhs, Variable<RHSValueType> rhs)
		: lhs_{lhs}, rhs_{rhs}
	{}
	
	binop_result_t get_value() 
	{
		LHSValueType lhs_value = lhs_.get_value();
		RHSValueType rhs_value = rhs_.get_value();
		return BinaryOp::evaluate(lhs_value, rhs_value);
	}

private:
	Variable<LHSValueType> lhs_;
	Variable<RHSValueType> rhs_;

};


template <class LHSValueType, class RHSValueType>
struct BinaryOp {
	
	using binop_result_t = std::common_type_t<
		typename node_traits<LHSValueType>::binop_result_t,
		typename node_traits<RHSValueType>::binop_result_t
			>;

	binop_result_t evaluate();

};

template <class LHSValueType, class RHSValueType>
struct AddOp : BinaryOp<LHSValueType, RHSValueType> {
	
	using binop_result_t = std::common_type_t<
		typename node_traits<LHSValueType>::binop_result_t,
		typename node_traits<RHSValueType>::binop_result_t
			>;

	static binop_result_t evaluate(LHSValueType x, RHSValueType y)
	{
		return x + y;
	}

};

template <class LHSValueType, class RHSValueType>
struct MultOp : BinaryOp<LHSValueType, RHSValueType> {
	
	using binop_result_t = std::common_type_t<
		typename node_traits<LHSValueType>::binop_result_t,
		typename node_traits<RHSValueType>::binop_result_t
			>;

	static binop_result_t evaluate(LHSValueType x, RHSValueType y)
	{
		return x * y;
	}

};

template <class LHSValueType, class RHSValueType>
inline constexpr auto operator+(
	const Variable<LHSValueType>& lhs,
	const Variable<RHSValueType>& rhs)
{
	return BinaryOpNode<Variable<LHSValueType>, Variable<RHSValueType>, AddOp<LHSValueType, RHSValueType> >(lhs, rhs);
}

template <class LHSValueType, class RHSValueType>
inline constexpr auto operator*(
	const Variable<LHSValueType>& lhs,
	const Variable<RHSValueType>& rhs)
{
	return BinaryOpNode<Variable<LHSValueType>, Variable<RHSValueType>, MultOp<LHSValueType, RHSValueType> >(lhs, rhs);
}

#ifdef AUTOPPL_USE_CONCEPTS
#endif

} // namespace ppl
