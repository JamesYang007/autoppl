#pragma once
#include <fastad_bits/reverse/core/sqrt.hpp>
#include <autoppl/util/traits/traits.hpp>

namespace ppl {
namespace expr {
namespace var {

/**
 * SqrtNode is a generic object representing sqrt on a variable expression.
 *
 * @tparam  VarExprType     variable expression type
 */

template <class VarExprType>
struct SqrtNode: 
    util::VarExprBase<SqrtNode<VarExprType>>
{
private:
    using expr_t = VarExprType;

	static_assert(util::is_var_expr_v<expr_t>);

public:
	using value_t = typename util::var_expr_traits<expr_t>::value_t;
    using shape_t = typename util::shape_traits<expr_t>::shape_t;
    static constexpr bool has_param = expr_t::has_param;

	SqrtNode(const expr_t& expr)
		: expr_{expr}
	{}

    template <class Func>
    void traverse(Func&&) const {}

    auto get() const { return eval_helper(expr_.get()); }
    auto eval() { return eval_helper(expr_.eval()); }
    
    constexpr size_t size() const { return expr_.size(); }
    constexpr size_t rows() const { return expr_.rows(); }
    constexpr size_t cols() const { return expr_.cols(); }

    template <class PtrPackType>
    auto ad(const PtrPackType& pack) const
    {  
        return ad::sqrt(expr_.ad(pack));
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
    auto eval_helper(const T& x) {
        if constexpr (util::is_scl_v<expr_t>) {
            return std::sqrt(x);
        } else {
            return x.array().sqrt().matrix();
        }
    }

	expr_t expr_;
};

} // namespace var
} // namespace expr

template <class ExprType
        , class = std::enable_if_t<
            util::is_valid_op_param_v<ExprType> &&
            !std::is_arithmetic_v<ExprType>
        > >
constexpr inline auto sqrt(const ExprType& expr)
{ 
    using expr_t = util::convert_to_param_t<ExprType>; 
    expr_t wrap_expr = expr; 
    using unary_t = expr::var::SqrtNode<expr_t>; 
    return unary_t(wrap_expr); 
}

} // namespace ppl
