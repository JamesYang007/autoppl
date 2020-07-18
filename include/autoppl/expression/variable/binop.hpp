#pragma once
#include <algorithm>
#include <autoppl/util/traits/var_expr_traits.hpp>
#include <autoppl/util/functional.hpp>

#define PPL_BINOP_EQUAL_FIXED_SIZE \
    "If both lhs and rhs are of fixed size, " \
    "then they must have the same size. "
#define PPL_BINOP_NO_MAT_SUPPORT \
    "Binary operations with matrices are not supported yet. "

namespace ppl {
namespace expr {

/**
 * BinaryOpNode is a generic object representing some binary operation
 * between two variable expressions.
 * For example, +,-,*,/ are four common binary operations.
 *
 * Currently binary operation with matrices is not supported.
 *
 * If both variable expressions are of fixed size, then it may
 * choose to perform some optimization, in which case, the size,
 * i.e. number of elements, has to be equal.
 *
 * @tparam  BinaryOp        binary operation policy containing a static member
 *                          function "evaluate(T x, T y)" that evaluates the
 *                          corresponding binary operation on the parameters.
 *                          See AddOp as an example below.
 * @tparam  LHSVarExprType  lhs variable expression type
 * @tparam  RHSVarExprType  rhs variable expression type
 */

template <class BinaryOp
        , class LHSVarExprType
        , class RHSVarExprType>
struct BinaryOpNode: 
    util::VarExprBase<BinaryOpNode<BinaryOp, LHSVarExprType, RHSVarExprType>>
{
	static_assert(util::is_var_expr_v<LHSVarExprType>);
	static_assert(util::is_var_expr_v<RHSVarExprType>);

    static_assert(!util::is_mat_v<LHSVarExprType> &&
                  !util::is_mat_v<RHSVarExprType>,
                  PPL_BINOP_NO_MAT_SUPPORT);

    static_assert(!util::is_fixed_size_v<LHSVarExprType> ||
                  !util::is_fixed_size_v<RHSVarExprType> ||
                  (util::var_expr_traits<LHSVarExprType>::fixed_size ==
                  util::var_expr_traits<RHSVarExprType>::fixed_size),
                  PPL_BINOP_EQUAL_FIXED_SIZE
                  );

	using value_t = std::common_type_t<
		typename util::var_expr_traits<LHSVarExprType>::value_t,
		typename util::var_expr_traits<RHSVarExprType>::value_t
			>;
    using shape_t = util::max_shape_t<
        typename util::shape_traits<LHSVarExprType>::shape_t,
        typename util::shape_traits<RHSVarExprType>::shape_t
            >;
    using index_t = uint32_t;

    static constexpr bool has_param = 
        LHSVarExprType::has_param || RHSVarExprType::has_param;

    static constexpr size_t fixed_size = 
        util::var_expr_traits<LHSVarExprType>::fixed_size;

	BinaryOpNode(const LHSVarExprType& lhs, 
                 const RHSVarExprType& rhs)
		: lhs_{lhs}, rhs_{rhs}
	{}

    template <class PVecType
            , class F = util::identity>
    value_t value(const PVecType& pvalues, 
                  size_t i,
                  F f = F()) const 
    {
        auto lhs_value = lhs_.value(pvalues, i, f);
        auto rhs_value = rhs_.value(pvalues, i, f);
        return BinaryOp::evaluate(lhs_value, rhs_value);
    }

    size_t size() const { return std::max(lhs_.size(), rhs_.size()); }

    /**
     * Returns ad expression of the binary operation.
     */
    template <class VecADVarType>
    auto to_ad(const VecADVarType& vars,
               const VecADVarType& cache,
               size_t i=0) const
    {  
        return BinaryOp::evaluate(lhs_.to_ad(vars, cache, i),
                                  rhs_.to_ad(vars, cache, i));
    }

    /**
     * Binop currently does not use any cache
     */
    index_t set_cache_offset(index_t idx) 
    { 
        idx = lhs_.set_cache_offset(idx);
        return rhs_.set_cache_offset(idx);
    }

private:
	LHSVarExprType lhs_;
	RHSVarExprType rhs_;
};

struct AddOp {
	template <class LHSValueType, class RHSValueType>
	static auto evaluate(LHSValueType x, RHSValueType y)
	{ return x + y; }
};

struct SubOp {
	template <class LHSValueType, class RHSValueType>
	static auto evaluate(LHSValueType x, RHSValueType y)
	{ return x - y; }
};

struct MultOp {
	template <class LHSValueType, class RHSValueType>
	static auto evaluate(LHSValueType x, RHSValueType y)
	{ return x * y; }
};

struct DivOp {
	template <class LHSValueType, class RHSValueType>
	static auto evaluate(LHSValueType x, RHSValueType y)
	{ return x / y; }
};

} // namespace expr
} // namespace ppl

#undef PPL_BINOP_EQUAL_FIXED_SIZE
#undef PPL_BINOP_NO_MAT_SUPPORT
