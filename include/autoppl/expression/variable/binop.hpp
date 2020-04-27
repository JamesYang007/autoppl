#pragma once
#include <autoppl/util/var_expr_traits.hpp>
#include <autoppl/variable.hpp>

#define MAX(a, b) ((a) > (b)) ? (a) : (b)

namespace ppl {
namespace expr {

template <class BinaryOp, class LHSVarExprType, class RHSVarExprType>
struct BinaryOpNode : 
    util::VarExpr<BinaryOpNode<BinaryOp, LHSVarExprType, RHSVarExprType>>
{
	static_assert(util::assert_is_var_expr_v<LHSVarExprType>);
	static_assert(util::assert_is_var_expr_v<RHSVarExprType>);

	using value_t = std::common_type_t<
		typename util::var_expr_traits<LHSVarExprType>::value_t,
		typename util::var_expr_traits<RHSVarExprType>::value_t
			>;

	BinaryOpNode(const LHSVarExprType& lhs, const RHSVarExprType& rhs)
		: lhs_{lhs}, rhs_{rhs}
	{ assert(lhs.size() == rhs.size() || lhs.size() == 1 || rhs.size() == 1); }

        value_t get_value(size_t i) const {
            auto lhs_value = lhs_.get_value(i);
            auto rhs_value = rhs_.get_value(i);
            return BinaryOp::evaluate(lhs_value, rhs_value);
        }

		size_t size() const { return MAX(lhs_.size(), rhs_.size()); }

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
