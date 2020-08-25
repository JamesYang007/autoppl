#pragma once
#include <fastad_bits/reverse/core/for_each.hpp>
#include <autoppl/util/traits/traits.hpp>

namespace ppl {
namespace expr {
namespace var {

/**
 * ForEachNode represents a for-loop where the expressions are contained in a vector.
 * It is a generalization of a GlueNode for variable expressions.
 *
 * @tparam VecExprType   type of vector of expressions 
 */

template <class VecExprType>
struct ForEachNode:
    util::VarExprBase<ForEachNode<VecExprType>>
{
private:
    using vec_expr_t = VecExprType;
    using elt_t = std::decay_t<typename VecExprType::value_type>;

    static_assert(util::is_var_expr_v<elt_t>);

public:
	using value_t = typename util::var_expr_traits<elt_t>::value_t;
    using shape_t = typename util::shape_traits<elt_t>::shape_t;
    static constexpr bool has_param = 
        util::var_expr_traits<elt_t>::has_param;

    ForEachNode(const vec_expr_t& vec_expr)
        : vec_expr_{vec_expr}
    {}

    template <class Func>
    void traverse(Func&& f)
    {
        for (auto& expr: vec_expr_) expr.traverse(f);
    }

    template <class Func>
    void traverse(Func&& f) const
    {
        for (const auto& expr: vec_expr_) expr.traverse(f);
    }

    auto get() const { 
        assert(!vec_expr_.empty());
        return vec_expr_.back().get(); 
    }

    auto eval() { 
        for (auto& expr : vec_expr_) expr.eval();
        return get();
    }
    
    constexpr size_t size() const { return vec_expr_.empty() ? 0 : vec_expr_[0].size(); }
    constexpr size_t rows() const { return vec_expr_.empty() ? 0 : vec_expr_[0].rows(); }
    constexpr size_t cols() const { return vec_expr_.empty() ? 0 : vec_expr_[0].cols(); }

    template <class PtrPackType>
    auto ad(const PtrPackType& pack) const
    {  
        return ad::for_each(vec_expr_.begin(),
                            vec_expr_.end(),
                            [&](const auto& expr) {
                                return expr.ad(pack);
                            });
    }

    template <class PtrPackType>
    void bind(const PtrPackType& pack)
    { 
        if constexpr (elt_t::has_param) {
            for (auto& expr : vec_expr_) expr.bind(pack);
        }
    }

    void activate_refcnt() const { 
        for (const auto& expr : vec_expr_) expr.activate_refcnt();
    }
    
private:
    vec_expr_t vec_expr_;
};

} // namespace var
} // namespace expr

template <class Iter
        , class F>
inline constexpr auto for_each(Iter begin,
                               Iter end,
                               F f)
{
    using iter_elt_t = std::decay_t<typename std::iterator_traits<Iter>::value_type>;
    using ret_t = std::invoke_result_t<F, iter_elt_t>;
    using expr_t = util::convert_to_param_t<ret_t>;
    std::vector<expr_t> exprs;
    exprs.reserve(std::distance(begin, end));

    std::for_each(begin, end, [&](auto&& x) { exprs.emplace_back(f(x)); });
    
    return expr::var::ForEachNode<std::vector<expr_t>>(exprs);
}

} // namespace ppl
