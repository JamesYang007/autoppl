#pragma once
#include <tuple>
#include <autoppl/expression/program/activate.hpp>
#include <autoppl/expression/program/init_params.hpp>
#include <autoppl/util/traits/traits.hpp>

namespace ppl {
namespace expr {
namespace prog {

template <class ModelExpr>
struct ProgramNodeBase
{
    using model_t = ModelExpr;
    ProgramNodeBase(const model_t& model)
        : model_(model)
    {}

    auto& get_model() { return model_; }
    const auto& get_model() const { return model_; }

protected:
    model_t model_;
};

/**
 * ProgramNode is a class that groups different types of expressions
 * to run sequentially like a program.
 * It is only specialized for when the expressions are:
 * - model expression
 * - variable expression | model expression
 *
 * All other specializations are disabled.
 *
 * @tparam  ExprTypes   list of expression types to execute in that order
 */
template <class TupExprType
        , class = void>
struct ProgramNode;

template <class ModelExpr>
struct ProgramNode<std::tuple<ModelExpr>,
                   std::enable_if_t<util::is_model_expr_v<ModelExpr> 
                > >:
    util::ProgramExprBase<
        ProgramNode<std::tuple<ModelExpr>,
                    std::enable_if_t<util::is_model_expr_v<ModelExpr>> > 
            >,
    ProgramNodeBase<ModelExpr>
{
    using base_t = ProgramNodeBase<ModelExpr>;
    using typename base_t::model_t;
    using base_t::model_;

    ProgramNode(const model_t& model)
        : base_t(model)
    {}

    auto log_pdf() { return model_.log_pdf(); }

    template <class PtrPackType>
    auto ad_log_pdf(const PtrPackType& pack) const {
        return model_.ad_log_pdf(pack);
    }   

    auto activate() const {
        auto res = expr::activate(model_);
        model_.activate_refcnt();
        return res;
    }

    template <class PtrPackType>
    void bind(const PtrPackType& pack) {
        model_.bind(pack);
    }

    template <class GenType>
    void init_params(GenType& gen,
                     bool prune = true,
                     double radius = 2.) {
        expr::init_params(*this, gen, prune, radius);  
    }
};

template <class TPExpr, class ModelExpr>
struct ProgramNode<std::tuple<TPExpr, ModelExpr>,
                   std::enable_if_t<
                        util::is_var_expr_v<TPExpr> &&
                        util::is_model_expr_v<ModelExpr> 
                > >:
    util::ProgramExprBase<
        ProgramNode<std::tuple<TPExpr, ModelExpr>,
                    std::enable_if_t<
                        util::is_var_expr_v<TPExpr> &&
                        util::is_model_expr_v<ModelExpr>> > 
            >,
    ProgramNodeBase<ModelExpr>
{
    using base_t = ProgramNodeBase<ModelExpr>;
    using tp_expr_t = TPExpr;
    using typename base_t::model_t;
    using base_t::model_;

    ProgramNode(const tp_expr_t& tp_expr,
                const model_t& model)
        : base_t(model)
        , tp_expr_(tp_expr)
    {}

    auto log_pdf() { 
        tp_expr_.eval();
        return model_.log_pdf(); 
    }

    template <class PtrPackType>
    auto ad_log_pdf(const PtrPackType& pack) const {
        return (tp_expr_.ad(pack), model_.ad_log_pdf(pack));
    }   

    auto activate() const {
        auto tp_res = expr::activate(tp_expr_);
        auto model_res = expr::activate(model_);
        tp_expr_.activate_refcnt();
        model_.activate_refcnt();

        util::OffsetPack cont_res;
        util::OffsetPack disc_res;
        cont_res = std::get<0>(model_res);
        cont_res.tp_offset = std::get<0>(tp_res).tp_offset;
        disc_res = std::get<1>(model_res);
        disc_res.tp_offset = std::get<1>(tp_res).tp_offset;
        return std::make_pair(cont_res, disc_res);
    }

    template <class PtrPackType>
    void bind(const PtrPackType& pack) {
        tp_expr_.bind(pack);
        model_.bind(pack);
    }

    template <class GenType>
    void init_params(GenType& gen,
                     bool prune = true,
                     double radius = 2.) {
        expr::init_params(*this, gen, prune, radius);  
    }

private:
    tp_expr_t tp_expr_;
};

} // namespace prog
} // namespace expr
} // namespace ppl
