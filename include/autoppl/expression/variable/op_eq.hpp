#pragma once
#include <autoppl/util/traits/traits.hpp>

namespace ppl {
namespace expr {
namespace var {

struct Eq
{
    template <class LHSType
            , class RHSType>
    constexpr static LHSType& eval(LHSType& lhs, const RHSType& rhs)
    { return lhs = rhs; }

    template <class LHSType
            , class RHSType>
    constexpr static auto eval(const LHSType& lhs, const RHSType& rhs)
    { return lhs = rhs; }
};

struct AddEq
{
    template <class LHSType
            , class RHSType>
    constexpr static LHSType& eval(LHSType& lhs, const RHSType& rhs)
    { return lhs += rhs; }

    template <class LHSType
            , class RHSType>
    constexpr static auto eval(const LHSType& lhs, const RHSType& rhs)
    { return lhs += rhs; }
};

struct SubEq
{
    template <class LHSType
            , class RHSType>
    constexpr static LHSType& eval(LHSType& lhs, const RHSType& rhs)
    { return lhs -= rhs; }

    template <class LHSType
            , class RHSType>
    constexpr static auto eval(const LHSType& lhs, const RHSType& rhs)
    { return lhs -= rhs; }
};

struct MulEq
{
    template <class LHSType
            , class RHSType>
    constexpr static LHSType& eval(LHSType& lhs, const RHSType& rhs)
    { return lhs *= rhs; }

    template <class LHSType
            , class RHSType>
    constexpr static auto eval(const LHSType& lhs, const RHSType& rhs)
    { return lhs *= rhs; }
};

struct DivEq
{
    template <class LHSType
            , class RHSType>
    constexpr static LHSType& eval(LHSType& lhs, const RHSType& rhs)
    { return lhs /= rhs; }

    template <class LHSType
            , class RHSType>
    constexpr static auto eval(const LHSType& lhs, const RHSType& rhs)
    { return lhs /= rhs; }
};

template <class Op
        , class TParamViewType
        , class VarExprType>
struct OpEqNode:
    util::VarExprBase<OpEqNode<Op, TParamViewType, VarExprType>>
{
private:
    using op_t = Op;
    using tp_view_t = TParamViewType;
    using var_expr_t = VarExprType;

	static_assert(util::is_tparam_v<tp_view_t>);
	static_assert(util::is_var_expr_v<var_expr_t>);

    static_assert(std::is_same_v<
            typename util::var_expr_traits<tp_view_t>::value_t,
            typename util::var_expr_traits<var_expr_t>::value_t >);

    static_assert(util::is_scl_v<var_expr_t> ||
            std::is_same_v<
                typename util::shape_traits<tp_view_t>::shape_t,
                typename util::shape_traits<var_expr_t>::shape_t >);

public:
	using value_t = typename util::var_expr_traits<tp_view_t>::value_t;
    using shape_t = typename util::shape_traits<tp_view_t>::shape_t;
    static constexpr bool has_param = true;

	OpEqNode(const tp_view_t& tp_view, 
             const var_expr_t& expr)
		: tp_view_{tp_view}, expr_{expr}
	{}

    template <class Func>
    void traverse(Func&& f)
    {
        static_cast<void>(f);
        if constexpr (std::is_same_v<Op, Eq>) {
            f(*this);
        }
    }

    template <class Func>
    void traverse(Func&& f) const
    {
        static_cast<void>(f);
        if constexpr (std::is_same_v<Op, Eq>) {
            f(*this);
        }
    }

    auto get() const { return tp_view_.get(); }

    auto eval() { 
        if constexpr (util::is_scl_v<tp_view_t> && 
                      util::is_scl_v<var_expr_t>) {
            return op_t::eval(tp_view_.get(), expr_.eval()); 
        } else if constexpr (!util::is_scl_v<tp_view_t> &&
                             util::is_scl_v<var_expr_t>) {
            auto tpa = tp_view_.get().array();
            return op_t::eval(tpa, expr_.eval());
        } else {
            auto tpa = tp_view_.get().array();
            return op_t::eval(tpa, expr_.eval().array());
        }
    }
    
    constexpr size_t size() const { return tp_view_.size(); }
    constexpr size_t rows() const { return tp_view_.rows(); }
    constexpr size_t cols() const { return tp_view_.cols(); }

    template <class PtrPackType>
    auto ad(const PtrPackType& pack) const
    {  
        return op_t::eval(tp_view_.ad(pack), expr_.ad(pack));
    }

    template <class PtrPackType>
    void bind(const PtrPackType& pack)
    { 
        if constexpr (tp_view_t::has_param) {
            tp_view_.bind(pack);
        }
        if constexpr (var_expr_t::has_param) {
            expr_.bind(pack);
        }
    }

    void activate_refcnt() const { 
        tp_view_.activate_refcnt();
        expr_.activate_refcnt(); 
    }

    auto& get_variable() { return tp_view_; }
    const auto& get_variable() const { return tp_view_; }

private:
    tp_view_t tp_view_;
    var_expr_t expr_;
};

namespace details {

template <class Op
        , class TParamViewType
        , class VarExprType>
constexpr inline auto opeq_helper(const TParamViewType& tp_view,
                                  const VarExprType& expr)
{ 
	using tp_view_t = util::convert_to_param_t<TParamViewType>;
    using expr_t = util::convert_to_param_t<VarExprType>;

   	tp_view_t wrap_tp_view = tp_view;
    expr_t wrap_expr = expr;

    return OpEqNode<Op, tp_view_t, expr_t>(wrap_tp_view, wrap_expr); 
}

} // namespace details
} // namespace var
} // namespace expr
} // namespace ppl
