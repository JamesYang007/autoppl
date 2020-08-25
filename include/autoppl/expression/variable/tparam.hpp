#pragma once
#include <autoppl/util/value.hpp>
#include <autoppl/util/traits/traits.hpp>
#include <autoppl/util/packs/offset_pack.hpp>
#include <autoppl/util/packs/ptr_pack.hpp>
#include <fastad_bits/reverse/core/var_view.hpp>

namespace ppl {

// forward declaration
namespace expr {
namespace var {
    
template <class Op
        , class TParamViewType
        , class VarExprType>
struct OpEqNode;
struct Eq;

} // namespace var
} // namespace expr

namespace details {

struct TParamInfoPack 
{
    util::OffsetPack off_pack;
};

} // namespace details

template <class Derived>
struct TParamViewBase;

template <class ValueType
        , class ShapeType>
struct TParamViewBase<TParamView<ValueType, ShapeType>>
{
    using value_t = ValueType;
    using shape_t = ShapeType;
    using derived_t = TParamView<value_t, shape_t>;
    using var_t = util::var_t<value_t, shape_t>;
    using id_t = const void*;
    static constexpr bool has_param = true;

    TParamViewBase(details::TParamInfoPack* i_pack,
                   size_t rows=1,
                   size_t cols=1) noexcept
        : i_pack_(i_pack)
        , var_(util::make_val<value_t, shape_t>(rows, cols))
        , id_(this)
    {}

    template <class VarExprType
            , class = std::enable_if_t<
                util::is_valid_op_param_v<VarExprType>
            > >
    auto operator=(const VarExprType& expr) const
    {
        using expr_t = util::convert_to_param_t<VarExprType>;
        expr_t wrap_expr = expr;
        return expr::var::OpEqNode<expr::var::Eq, derived_t, expr_t>(
                static_cast<const derived_t&>(*this), wrap_expr); 
    }

    template <class Func>
    void traverse(Func&&) const {}

    const var_t& eval() { return get(); }

    template <class UCValPtrType
            , class UCAdjPtrType
            , class CValPtrType>
    auto ad(const util::PtrPack<UCValPtrType, 
                                UCAdjPtrType, 
                                value_t*,
                                value_t*,
                                CValPtrType>& pack) const { 
        return ad::VarView<value_t, shape_t>(pack.tp_val + i_pack_->off_pack.tp_offset, 
                                             pack.tp_adj + i_pack_->off_pack.tp_offset,
                                             rows(), cols());
    }
    
    void activate(util::OffsetPack& pack) const { 
        i_pack_->off_pack = pack;
        pack.tp_offset += size();
    }

    void activate_refcnt() const {}

    template <class PtrPackType>
    void bind(const PtrPackType& pack) 
    { 
        static_cast<void>(pack);
        if constexpr (std::is_convertible_v<typename PtrPackType::tp_val_ptr_t, value_t*>) {
            value_t* tcp = pack.tp_val;
            util::bind(var_, tcp + i_pack_->off_pack.tp_offset, 
                       rows(), cols());
        }
    }

    var_t& get() { return util::get(var_); }
    const var_t& get() const { return util::get(var_); }
    constexpr size_t size() const { return util::size(var_); }
    constexpr size_t rows() const { return util::rows(var_); }
    constexpr size_t cols() const { return util::cols(var_); }
    id_t id() const { return id_; }

protected:
    using view_t = ad::util::shape_to_raw_view_t<value_t, shape_t>;
    details::TParamInfoPack* const i_pack_;
    view_t var_;
    const id_t id_; 
};

template <class ValueType
        , class ShapeType = ppl::scl>
struct TParamView:
    TParamViewBase<TParamView<ValueType, ShapeType>>,
    util::VarExprBase<TParamView<ValueType, ShapeType>>,
    util::TParamBase<TParamView<ValueType, ShapeType>> 
{
    using base_t = TParamViewBase<TParamView<ValueType, ShapeType>>;
    using base_t::operator=;

    TParamView(details::TParamInfoPack* i_pack,
               size_t rows=1,
               size_t cols=1) noexcept
        : base_t(i_pack, rows, cols)
    {}
};

template <class ValueType>
struct TParamView<ValueType, scl>:
    TParamViewBase<TParamView<ValueType, scl>>,
    util::VarExprBase<TParamView<ValueType, scl>>,
    util::TParamBase<TParamView<ValueType, scl>> 
{
    using base_t = TParamViewBase<TParamView<ValueType, scl>>;
    using base_t::operator=;

    TParamView(details::TParamInfoPack* i_pack,
               size_t rows=1,
               size_t cols=1,
               size_t rel_offset = 0) noexcept
        : base_t(i_pack, rows, cols)
        , rel_offset_(rel_offset)
    {}

    template <class UCValPtrType
            , class UCAdjPtrType
            , class CValPtrType>
    auto ad(const util::PtrPack<UCValPtrType, 
                                UCAdjPtrType, 
                                typename base_t::value_t*,
                                typename base_t::value_t*,
                                CValPtrType>& pack) const { 
        base_t::i_pack_->off_pack.tp_offset += rel_offset_;
        auto&& res = base_t::ad(pack);
        base_t::i_pack_->off_pack.tp_offset -= rel_offset_;
        return res;
    }

    template <class PtrPackType>
    void bind(const PtrPackType& pack) 
    { 
        base_t::i_pack_->off_pack.tp_offset += rel_offset_;
        base_t::bind(pack);
        base_t::i_pack_->off_pack.tp_offset -= rel_offset_;
    }

private:
    size_t rel_offset_;
};

template <class ValueType>
struct TParamView<ValueType, vec>:
    TParamViewBase<TParamView<ValueType, vec>>,
    util::VarExprBase<TParamView<ValueType, vec>>,
    util::TParamBase<TParamView<ValueType, vec>> 
{
    using base_t = TParamViewBase<TParamView<ValueType, vec>>;
    using base_t::operator=;

    TParamView(details::TParamInfoPack* i_pack,
               size_t rows,
               size_t cols=1) noexcept
        : base_t(i_pack, rows, cols)
    {}

    auto operator[](size_t i) const {
        return TParamView<typename base_t::value_t, scl>(
                base_t::i_pack_, base_t::rows(), base_t::cols(), i);
    }
};

template <class ValueType
        , class ShapeType = ppl::scl>
struct TParam;

template <class ValueType>
struct TParam<ValueType, ppl::scl>: 
    TParamView<ValueType, ppl::scl>,
    util::TParamBase<TParam<ValueType, ppl::scl>>
{
    using base_t = TParamView<ValueType, ppl::scl>;
    using base_t::operator=;

    TParam() noexcept
        : base_t(&i_pack_, 1, 1)
    {}

private:
    details::TParamInfoPack i_pack_;
};

template <class ValueType>
struct TParam<ValueType, ppl::vec> : 
    TParamView<ValueType, ppl::vec>,
    util::TParamBase<TParam<ValueType, ppl::vec>>
{
    using base_t = TParamView<ValueType, ppl::vec>;
    using typename base_t::value_t;
    using base_t::operator=;

    TParam(size_t n)
        : base_t(&i_pack_, n, 1)
    {}

private:
    details::TParamInfoPack i_pack_;
};

template <class ValueType>
struct TParam<ValueType, ppl::mat> : 
    TParamView<ValueType, ppl::mat>,
    util::TParamBase<TParam<ValueType, ppl::mat>>
{
    using base_t = TParamView<ValueType, ppl::mat>;
    using base_t::operator=;

    TParam(size_t rows, size_t cols)
        : base_t(&i_pack_, rows, cols)
    {}

private:
    details::TParamInfoPack i_pack_;
};

} // namespace ppl
