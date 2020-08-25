#pragma once
#include <fastad_bits/reverse/core/math.hpp>
#include <autoppl/util/traits/traits.hpp>

namespace ppl {
namespace expr {
namespace var {

/**
 * UnaryNode is a generic object representing some unary operation on a variable expression.
 * For example, log, exp, sin, cos, tan are common unary operations.
 *
 * @tparam  UnaryOp         unary operation policy containing a static member
 *                          function "fmap(T x)" that evaluates the
 *                          corresponding unary operation on the underlying expression.
 * @tparam  VarExprType     variable expression type
 */

template <class UnaryOp
        , class VarExprType>
struct UnaryNode: 
    util::VarExprBase<UnaryNode<UnaryOp, VarExprType>>
{
private:
    using expr_t = VarExprType;

	static_assert(util::is_var_expr_v<expr_t>);

public:
	using value_t = typename util::var_expr_traits<expr_t>::value_t;
    using shape_t = typename util::shape_traits<expr_t>::shape_t;
    static constexpr bool has_param = expr_t::has_param;

	UnaryNode(const expr_t& expr)
		: expr_{expr}
	{}

    template <class Func>
    void traverse(Func&&) const {}

    auto get() const { 
        return eval_helper(expr_.get());
    }

    auto eval() {
        return eval_helper(expr_.eval());
    }
    
    constexpr size_t size() const { return expr_.size(); }
    constexpr size_t rows() const { return expr_.rows(); }
    constexpr size_t cols() const { return expr_.cols(); }

    template <class PtrPackType>
    auto ad(const PtrPackType& pack) const
    {  
        return UnaryOp::fmap(expr_.ad(pack));
    }

    template <class PtrPackType>
    void bind(const PtrPackType& pack)
    { 
        if constexpr (expr_t::has_param) {
            expr_.bind(pack);
        }
    }

    void activate_refcnt() const { 
        expr_.activate_refcnt();
    }

private:

    template <class T>
    auto eval_helper(const T& x) const {
        if constexpr (util::is_scl_v<expr_t>) {
            return UnaryOp::fmap(x);
        } else {
            return UnaryOp::fmap(x.array()).matrix();
        }
    }

	expr_t expr_;
};

} // namespace var
} // namespace expr

#define PPL_UNARY_FUNC(name, strct) \
    template <class ExprType    \
            , class = std::enable_if_t< \
                util::is_valid_op_param_v<ExprType> && \
                !std::is_arithmetic_v<ExprType> \
            > > \
    constexpr inline auto name(const ExprType& expr) \
    {   \
        using expr_t = util::convert_to_param_t<ExprType>;  \
        expr_t wrap_expr = expr;    \
        using unary_t = expr::var::UnaryNode<ad::math::strct, expr_t>;  \
        return unary_t(wrap_expr);  \
    }

PPL_UNARY_FUNC(sin, Sin)
PPL_UNARY_FUNC(cos, Cos)
PPL_UNARY_FUNC(tan, Tan)
PPL_UNARY_FUNC(asin, Arcsin)
PPL_UNARY_FUNC(acos, Arccos)
PPL_UNARY_FUNC(atan, Arctan)
PPL_UNARY_FUNC(exp, Exp)
PPL_UNARY_FUNC(log, Log)

} // namespace ppl
