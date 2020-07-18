#pragma once
#include <array>
#include <fastad_bits/node.hpp>
#include <autoppl/util/traits/var_expr_traits.hpp>
#include <autoppl/util/functional.hpp>
#include <autoppl/util/iterator/counting_iterator.hpp>

#define PPL_DOT_MAT_VEC \
    "Dot product is only supported for matrix as lhs argument " \
    "and a vector as rhs argument. "

namespace ppl {
namespace expr {

/**
 * This class represents a dot product between a matrix
 * expression and a vector expression.
 * No other combination of shapes is allowed to be represented currently
 * (compiler error if user attempts to pass in other shapes).
 * 
 * This expression is currently not optimized for fixed-size matrix
 * AND fixed-size vector - it is always assumed to be sized dynamically.
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

public:
	static_assert(util::is_var_expr_v<lhs_t>);
	static_assert(util::is_var_expr_v<rhs_t>);
    static_assert(util::is_mat_v<lhs_t> &&
                  util::is_vec_v<rhs_t>, 
                  PPL_DOT_MAT_VEC);

	using value_t = std::common_type_t<
		typename util::var_expr_traits<lhs_t>::value_t,
		typename util::var_expr_traits<rhs_t>::value_t
			>;
    using shape_t = ppl::vec;
    using index_t = uint32_t;

    static constexpr bool has_param = 
        lhs_t::has_param || rhs_t::has_param;

    // currently set to 0 to force-treat as non-fixed size
    static constexpr size_t fixed_size = 0;

	DotNode(const lhs_t& lhs, 
            const rhs_t& rhs)
		: lhs_{lhs}
        , rhs_{rhs}
    {}

    template <class PVecType
            , class F = util::identity>
    value_t value(const PVecType& pvalues, 
                  size_t i,
                  F f = F()) const 
    {
        value_t dot = 0;
        for (size_t j = 0; j < rhs_.size(); ++j) {
            dot += lhs_.value(pvalues, i, j, f) *
                    rhs_.value(pvalues, j, f); 
        }
        return dot;
    }

    size_t size() const { return lhs_.nrows(); }

    /**
     * Returns ad expression of the dot-product for ith element.
     *
     * NOTES: 
     *
     * - only defined behavior when user can guarantee that first element
     *   is computed before any other element. If so, order
     *   of evaluation for other elements does not matter.
     *
     * - user must guarantee that if there are multiple AD expressions built
     *   from this object and sharing the same cache, the cache adjoints are reset
     *   after each backward evaluation of the expressions.
     *   Forward evaluations do not require any resets.
     *   
     * - user cannot forward evaluate one expr, forward evaluate another,
     *   then reverse evaluate the former, since the second forward evaluation
     *   will have overwritten the cache variables.
     */
    template <class VecADVarType>
    auto to_ad(const VecADVarType& vars,
               const VecADVarType& cache,
               size_t i) const
    {  

        auto to_glue = [&](auto k) { 
            return (cache[offset_+k] = 
                    rhs_.to_ad(vars, cache, k));
        };
        auto fev = (i == 0) ? ad::for_each(
                                util::counting_iterator<>(0),
                                util::counting_iterator<>(rhs_.size()),
                                to_glue) :
                              ad::for_each(
                                util::counting_iterator<>(0),
                                util::counting_iterator<>(0),
                                to_glue);

        return (fev,
                ad::sum(util::counting_iterator<>(0),
                        util::counting_iterator<>(rhs_.size()),
                        [&, i](auto j) {
                           return lhs_.to_ad(vars, cache, i, j) *
                                   cache[offset_+j];
                        })
                );

    }

    /**
     * Requires vector (RHS) length number of AD variables from cache.
     * Each AD variable will cache the results for rhs's expression evaluations.
     */
    index_t set_cache_offset(index_t offset) 
    {
        offset = lhs_.set_cache_offset(offset);
        offset = rhs_.set_cache_offset(offset);
        offset_ = offset;
        return offset_ + rhs_.size(); 
    }

private:
    lhs_t lhs_;
    rhs_t rhs_;
    index_t offset_;
};

} // namespace expr
} // namespace ppl

#undef PPL_DOT_MAT_VEC
