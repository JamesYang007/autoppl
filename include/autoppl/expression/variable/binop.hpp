#pragma once

#include <type_traits>
#include <autoppl/util/traits.hpp>
#include <autoppl/variable.hpp>

namespace ppl {
namespace expr {

template <class BinaryOp, class LHSVarExprType, class RHSVarExprType>
struct BinaryOpNode
{
	static_assert(util::assert_is_var_expr_v<LHSVarExprType>);
	static_assert(util::assert_is_var_expr_v<RHSVarExprType>);

	using value_t = std::common_type_t<
		typename util::var_expr_traits<LHSVarExprType>::value_t,
		typename util::var_expr_traits<RHSVarExprType>::value_t
			>;

	BinaryOpNode(const LHSVarExprType& lhs, const RHSVarExprType& rhs)
		: lhs_{lhs}, rhs_{rhs}
	{}
	
	value_t get_value() const
	{
		auto lhs_value = lhs_.get_value();
		auto rhs_value = rhs_.get_value();
		return BinaryOp::evaluate(lhs_value, rhs_value);
	}

	explicit operator value_t () const 
	{
		return get_value();
	}

private:
	LHSVarExprType lhs_;
	RHSVarExprType rhs_;

};

struct AddOp {
	
	template <class LHSValueType, class RHSValueType>
	static auto evaluate(LHSValueType x, RHSValueType y)
	{
		return x + y;
	}

};

struct SubOp {
	
	template <class LHSValueType, class RHSValueType>
	static auto evaluate(LHSValueType x, RHSValueType y)
	{
		return x - y;
	}

};

struct MultOp {
	
	template <class LHSValueType, class RHSValueType>
	static auto evaluate(LHSValueType x, RHSValueType y)
	{
		return x * y;
	}

};

struct DivOp {
	
	template <class LHSValueType, class RHSValueType>
	static auto evaluate(LHSValueType x, RHSValueType y)
	{
		return x / y;
	}

};

} // namespace expr
} // namespace ppl
