#pragma once
#include <autoppl/util/traits/traits.hpp>

namespace ppl {
namespace expr {
namespace var {

template <class LHSExprType
        , class RHSExprType>
struct GlueNode:
    util::VarExprBase<GlueNode<LHSExprType, RHSExprType>>
{
private:
    using lhs_t = LHSExprType;
    using rhs_t = RHSExprType;

    static_assert(util::is_var_expr_v<lhs_t>);
    static_assert(util::is_var_expr_v<rhs_t>);

public:
	using value_t = typename util::var_expr_traits<rhs_t>::value_t;
    using shape_t = typename util::shape_traits<rhs_t>::shape_t;
    static constexpr bool has_param = 
        util::var_expr_traits<lhs_t>::has_param ||
        util::var_expr_traits<rhs_t>::has_param;

    GlueNode(const lhs_t& lhs,
             const rhs_t& rhs)
        : lhs_{lhs}, rhs_{rhs}
    {}

    template <class Func>
    void traverse(Func&& f)
    {
        lhs_.traverse(f);
        rhs_.traverse(f);
    }

    template <class Func>
    void traverse(Func&& f) const
    {
        lhs_.traverse(f);
        rhs_.traverse(f);
    }

    auto get() const { return rhs_.get(); }

    auto eval() { 
        lhs_.eval();
        return rhs_.eval();
    }
    
    constexpr size_t size() const { return rhs_.size(); }
    constexpr size_t rows() const { return rhs_.rows(); }
    constexpr size_t cols() const { return rhs_.cols(); }

    template <class PtrPackType>
    auto ad(const PtrPackType& pack) const
    {  
        return (lhs_.ad(pack), rhs_.ad(pack));
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
} // namespace ppl
