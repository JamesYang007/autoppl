#pragma once
#include <fastad_bits/reverse/core/dot.hpp>
#include <autoppl/util/traits/traits.hpp>

#define PPL_DOT_MAT_VEC \
    "Dot product is only supported for matrix as lhs argument " \
    "and a matrix or vector as rhs argument. "

namespace ppl {
namespace expr {
namespace var {

/**
 * This class represents a dot product between a matrix
 * expression and a vector expression.
 * No other combination of shapes is allowed to be represented currently
 * (compiler error if user attempts to pass in other shapes).
 *
 * @tparam LHSVarExprType   lhs variable expression type
 * @tparam RHSVarExprType   rhs variable expression type
 */

template <class LHSVarExprType
        , class RHSVarExprType>
class DotNode:
    util::VarExprBase<DotNode<LHSVarExprType, RHSVarExprType>>
{
    using lhs_t = LHSVarExprType;
    using rhs_t = RHSVarExprType;

	static_assert(util::is_var_expr_v<lhs_t>);
	static_assert(util::is_var_expr_v<rhs_t>);
    static_assert(util::is_mat_v<lhs_t> &&
                  (util::is_vec_v<rhs_t> || util::is_mat_v<rhs_t>), 
                  PPL_DOT_MAT_VEC);

public:
	using value_t = std::common_type_t<
		typename util::var_expr_traits<lhs_t>::value_t,
		typename util::var_expr_traits<rhs_t>::value_t
			>;
    using shape_t = ad::core::details::dot_shape_t<lhs_t, rhs_t>;
    static constexpr bool has_param = 
        lhs_t::has_param || rhs_t::has_param;

	DotNode(const lhs_t& lhs, 
            const rhs_t& rhs)
		: lhs_{lhs}
        , rhs_{rhs}
    {}

    template <class Func>
    void traverse(Func&&) const {}

    auto eval() { return lhs_.eval() * rhs_.eval(); }
    auto get() { return lhs_.get() * rhs_.get(); }
    size_t size() const { return rows() * cols(); }
    size_t rows() const { return lhs_.rows(); }
    size_t cols() const { return rhs_.cols(); }

    template <class PtrPackType>
    auto ad(const PtrPackType& pack) const
    {  
        return ad::dot(lhs_.ad(pack), 
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
    lhs_t lhs_;
    rhs_t rhs_;
};

} // namespace var
} // namespace expr

/**
 * Builds a dot product expression for two expressions.
 */
template <class LHSVarExprType
        , class RHSVarExprType
        , class = std::enable_if_t<
            (util::is_var_v<LHSVarExprType> ||
             util::is_var_expr_v<LHSVarExprType>) &&
            (util::is_var_v<RHSVarExprType> ||
             util::is_var_expr_v<RHSVarExprType>)
        > >
inline constexpr auto dot(const LHSVarExprType& lhs,
                          const RHSVarExprType& rhs)
{
	using lhs_t = util::convert_to_param_t<LHSVarExprType>;
    using rhs_t = util::convert_to_param_t<RHSVarExprType>;

   	lhs_t wrap_lhs_expr = lhs;
    rhs_t wrap_rhs_expr = rhs;
    
    return expr::var::DotNode<lhs_t, rhs_t>(
            wrap_lhs_expr, wrap_rhs_expr);
}

} // namespace ppl

#undef PPL_DOT_MAT_VEC
