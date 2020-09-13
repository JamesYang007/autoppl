#pragma once
#include <fastad_bits/reverse/core/binary.hpp>
#include <autoppl/util/traits/traits.hpp>

#define PPL_BINOP_EQUAL_FIXED_SIZE \
    "If both lhs and rhs are of fixed size, " \
    "then they must have the same size. "
#define PPL_BINOP_NO_MAT_SUPPORT \
    "Binary operations with matrices are not supported yet. "

namespace ppl {
namespace expr {
namespace var {

/**
 * BinaryNode is a generic object representing some binary operation
 * between two variable expressions.
 * For example, +,-,*,/ are four common binary operations.
 *
 * If both variable expressions are of fixed size, then it may
 * choose to perform some optimization, in which case, the size,
 * i.e. number of elements, has to be equal.
 *
 * @tparam  BinaryOp        binary operation policy containing a static member
 *                          function "fmap(T x, U y)" that evaluates the
 *                          corresponding binary operation on the parameters.
 * @tparam  LHSVarExprType  lhs variable expression type
 * @tparam  RHSVarExprType  rhs variable expression type
 */

template <class BinaryOp
        , class LHSVarExprType
        , class RHSVarExprType>
struct BinaryNode: 
    util::VarExprBase<BinaryNode<BinaryOp, LHSVarExprType, RHSVarExprType>>
{
private:
    using lhs_t = LHSVarExprType;
    using rhs_t = RHSVarExprType;

	static_assert(util::is_var_expr_v<lhs_t>);
	static_assert(util::is_var_expr_v<rhs_t>);

public:
	using value_t = std::common_type_t<
		typename util::var_expr_traits<lhs_t>::value_t,
		typename util::var_expr_traits<rhs_t>::value_t
			>;
    using shape_t = ad::util::max_shape_t<
        typename util::shape_traits<lhs_t>::shape_t,
        typename util::shape_traits<rhs_t>::shape_t
            >;
    static constexpr bool has_param = 
        lhs_t::has_param || rhs_t::has_param;

	BinaryNode(const lhs_t& lhs, 
               const rhs_t& rhs)
		: lhs_{lhs}, rhs_{rhs}
	{}

    template <class Func>
    void traverse(Func&&) const {}

    auto get() const { 
        auto&& lhs = lhs_.get();
        auto&& rhs = rhs_.get();
        return eval_helper(lhs, rhs);
    }

    auto eval() 
    {
        auto&& lhs = lhs_.eval();
        auto&& rhs = rhs_.eval();
        return eval_helper(lhs, rhs);
    }
    
    size_t size() const { return std::max(lhs_.size(), rhs_.size()); }
    size_t rows() const { return std::max(lhs_.rows(), rhs_.rows()); }
    size_t cols() const { return std::max(lhs_.cols(), rhs_.cols()); }

    template <class PtrPackType>
    auto ad(const PtrPackType& pack) const
    {  
        return BinaryOp::fmap(lhs_.ad(pack),
                              rhs_.ad(pack));
    }

    template <class PtrPackType>
    void bind(const PtrPackType& pack)
    { 
        if constexpr (lhs_t::has_param) {
            lhs_.bind(pack);
        }
        if constexpr (rhs_t::has_param) {
            rhs_.bind(pack);
        }
    }

    void activate_refcnt() const { 
        lhs_.activate_refcnt();
        rhs_.activate_refcnt();
    }

private:

    template <class LHSType, class RHSType>
    auto eval_helper(const LHSType& lhs, 
                     const RHSType& rhs) const {
        if constexpr (util::is_scl_v<lhs_t> &&
                      util::is_scl_v<rhs_t>) {
            return BinaryOp::fmap(lhs, rhs);
        } else if constexpr (util::is_scl_v<lhs_t>) {
            return BinaryOp::fmap(lhs, rhs.array()).matrix();
        } else if constexpr (util::is_scl_v<rhs_t>) {
            return BinaryOp::fmap(lhs.array(), rhs).matrix();
        } else {
            return BinaryOp::fmap(lhs.array(), rhs.array()).matrix();
        }
    }

	lhs_t lhs_;
	rhs_t rhs_;
};

namespace details {

template <class Op, class LHSType, class RHSType>
inline constexpr auto operator_helper(const LHSType& lhs, 
                                      const RHSType& rhs)
{
    // note: may be reference types if converted to itself
	using lhs_t = util::convert_to_param_t<LHSType>;
    using rhs_t = util::convert_to_param_t<RHSType>;

   	lhs_t wrap_lhs_expr = lhs;
    rhs_t wrap_rhs_expr = rhs;
    
    using binary_t = BinaryNode<Op, lhs_t, rhs_t>;

	return binary_t(wrap_lhs_expr, wrap_rhs_expr);
}

} // namespace details
} // namespace var
} // namespace expr
} // namespace ppl

#undef PPL_BINOP_EQUAL_FIXED_SIZE
#undef PPL_BINOP_NO_MAT_SUPPORT
