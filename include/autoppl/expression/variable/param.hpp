#pragma once
#include <autoppl/expression/constraint/unconstrained.hpp>
#include <autoppl/util/traits/var_traits.hpp>
#include <autoppl/util/traits/shape_traits.hpp>
#include <autoppl/util/packs/offset_pack.hpp>
#include <autoppl/util/packs/ptr_pack.hpp>
#include <fastad_bits/reverse/core/var_view.hpp>

#define PPL_PARAMVIEW_SHAPE_UNSUPPORTED \
    "Unsupported shape for ParamView. "
#define PPL_PARAM_SHAPE_UNSUPPORTED \
    "Unsupported shape for Param. "

namespace ppl {
namespace details {

struct ParamInfoPack 
{
    size_t refcnt = 0;          // total reference count
    util::OffsetPack off_pack;
};

} // namespace details

/**
 * ParamView is a class that views or references a parameter entity.
 * It views the internal data of the first parameter object that was created.
 * This is our way of making sure that all references to a parameter object
 * when creating a model expression indeed refers to that first object.
 *
 * Users will likely not need to create these objects directly.
 * The easier-to-use Param class template will likely be used.
 * When constructing a model expression, both Param and ParamView objects 
 * will be converted to a ParamView.
 *
 * @tparam  ValueType       value type to view  
 * @tparam  ShapeType       shape of the object it is viewing. 
 * @tparam  ConstraintType  constraint expression type   
 */

template <class ValueType
        , class ShapeType = ppl::scl
        , class ConstraintType = expr::constraint::Unconstrained>
struct ParamView:
    util::VarExprBase<ParamView<ValueType, ShapeType, ConstraintType>>,
    util::ParamBase<ParamView<ValueType, ShapeType, ConstraintType>> 
{
    using value_t = ValueType;
    using shape_t = ShapeType;
    using constraint_t = ConstraintType;
    using var_t = util::var_t<value_t, shape_t>;
    using id_t = const void*;
    static constexpr bool has_param = true;

    ParamView(details::ParamInfoPack* i_pack,
              size_t rows=1,
              size_t cols=1,
              constraint_t c = constraint_t()) noexcept
        : i_pack_{i_pack}
        , transformer_{rows, cols, c}
        , id_{this}
    {}

    template <class Func>
    void traverse(Func&&) const {}

    /**
     * Evaluates the ParamView expression by first incrementing the visit count.
     * If it is the first to visit such parameter when evaluating the model,
     * it must first transform the unconstrained parameters to constrained parameters.
     * If it is the last to visit such parameter, it must reset the visit count back to 0
     * for the next time we evaluate the model.
     *
     * @return  constrained parameter view
     */
    const var_t& eval() { 
        transformer_.inv_transform(i_pack_->refcnt);
        return transformer_.get_c();
    }

    /**
     * This method is currently only used during pruning in initialization.
     * It assumes that reference counting is not needed, 
     * since a full model evaluation is not required when this is called.
     */
    void inv_eval() { transformer_.transform(); }

    template <class TPValPtrType
            , class TPAdjPtrType>
    auto ad(const util::PtrPack<value_t*, 
                                value_t*, 
                                TPValPtrType,
                                TPAdjPtrType,
                                value_t*>& pack) const { 
        auto curr_pack = pack;
        curr_pack.uc_val += i_pack_->off_pack.uc_offset; 
        curr_pack.uc_adj += i_pack_->off_pack.uc_offset;
        curr_pack.c_val += i_pack_->off_pack.c_offset; 
        curr_pack.v_val += i_pack_->off_pack.v_offset;
        return transformer_.inv_transform_ad(curr_pack, pack,
                                             i_pack_->refcnt);
    }

    template <class TPValPtrType
            , class TPAdjPtrType>
    auto logj_ad(const util::PtrPack<value_t*, 
                                     value_t*, 
                                     TPValPtrType,
                                     TPAdjPtrType,
                                     value_t*>& pack) const { 
        auto curr_pack = pack;
        curr_pack.uc_val += i_pack_->off_pack.uc_offset;
        curr_pack.uc_adj += i_pack_->off_pack.uc_offset;
        curr_pack.c_val += i_pack_->off_pack.c_offset;
        curr_pack.v_val += i_pack_->off_pack.v_offset;
        return transformer_.logj_inv_transform_ad(curr_pack, pack);
    }

    /**
     * Initialize unconstrained values by generating constrained values
     * and then transforming to unconstrained values.
     * Undefined behavior if bind has not been called before.
     */
    template <class GenType, class ContDist>
    void init(GenType& gen, ContDist& dist)
    { transformer_.init(gen, dist); }

    /**
     * Set the common offsets with pack and resets reference count to 0.
     * Updates pack to contain the next offsets after accounting for current ParamView.
     *
     * Note: this should only be called exactly once per parameter referenced in a model.
     * Since a model must assign a distribution to every parameter referenced in the model,
     * it suffices to activate parameters in those expressions.
     */
    void activate(util::OffsetPack& pack) const { 
        i_pack_->off_pack = pack;
        i_pack_->refcnt = 0;
        pack.uc_offset += transformer_.bind_size_uc();
        pack.c_offset += transformer_.bind_size_c();
        pack.v_offset += transformer_.bind_size_v();
    }

    /**
     * Increments the reference count to activate the current ParamView.
     * The total number of reference counts for a parameter is defined to be
     * the number of inverse-transforms required for the paarameter during model evaluation.
     * This is identical to the number of ParamView objects referencing the parrameter in the model.
     * In general, constraint expression may need to be activated as well.
     * It requires the current reference count to determine if the activation is needed.
     */
    void activate_refcnt() const { 
        ++i_pack_->refcnt; 
        transformer_.activate_refcnt(i_pack_->refcnt);
    }

    /**
     * Finds the correct offsetted pointers for the three parameters
     * using the common offsets in info pack and delegates binding to underlying transformer.
     * This only needs to be called if user wishes to call eval().
     * For AD support only, this does not need to be called.
     */
    template <class PtrPackType>
    void bind(const PtrPackType& pack) 
    { 
        static_cast<void>(pack);
        if constexpr (std::is_convertible_v<typename PtrPackType::uc_val_ptr_t, value_t*> &&
                      std::is_convertible_v<typename PtrPackType::c_val_ptr_t, value_t*>) {
            value_t* ucp = pack.uc_val;
            value_t* cp = pack.c_val;
            size_t* vp = pack.v_val;
            util::PtrPack curr_pack(ucp, nullptr, nullptr, nullptr, cp, vp);
            curr_pack.uc_val += i_pack_->off_pack.uc_offset;
            curr_pack.c_val += i_pack_->off_pack.c_offset; 
            curr_pack.v_val += i_pack_->off_pack.v_offset;
            transformer_.bind(curr_pack, pack);
        }
    }

    var_t& get() { return transformer_.get_c(); }
    const var_t& get() const { return transformer_.get_c(); }
    constexpr size_t size() const { return transformer_.size_c(); }
    constexpr size_t rows() const { return transformer_.rows_c(); }
    constexpr size_t cols() const { return transformer_.cols_c(); }
    id_t id() const { return id_; }

    // API specific to ParamView
    auto& offset() { return i_pack_->off_pack; }
    auto offset() const { return i_pack_->off_pack; }
    constexpr size_t size_uc() const { return transformer_.size_uc(); }
    constexpr size_t size_c() const { return transformer_.size_c(); }

private:
    details::ParamInfoPack* const i_pack_;
    expr::constraint::Transformer<value_t, shape_t, constraint_t> transformer_;
    const id_t id_; 
};

/**
 * Param is a class template wrapping a ParamView for user-friendly usage.
 * A Param is a ParamView (it views itself).
 * Similar to ParamView, it must be given a shape tag.
 *
 * @tparam ValueType    underlying value type (usually double or int)
 * @tparam ShapeType    one of the three shape tags.
 */

template <class ValueType
        , class ShapeType = ppl::scl
        , class ConstraintType = expr::constraint::Unconstrained>
struct Param
{
    static_assert(util::is_shape_v<ShapeType>,
                  PPL_PARAM_SHAPE_UNSUPPORTED);
};

template <class ValueType, class ConstraintType>
struct Param<ValueType, ppl::scl, ConstraintType>: 
    ParamView<ValueType, ppl::scl, ConstraintType>,
    util::ParamBase<Param<ValueType, ppl::scl, ConstraintType>>
{
    using base_t = ParamView<ValueType, ppl::scl, ConstraintType>;
    using typename base_t::constraint_t;

    Param(size_t=1,
          size_t=1,
          constraint_t c = constraint_t()) noexcept
        : base_t(&i_pack_, 1, 1, c)
    {}

private:
    details::ParamInfoPack i_pack_;
};

template <class ValueType, class ConstraintType>
struct Param<ValueType, ppl::vec, ConstraintType> : 
    ParamView<ValueType, ppl::vec, ConstraintType>,
    util::ParamBase<Param<ValueType, ppl::vec, ConstraintType>>
{
    using base_t = ParamView<ValueType, ppl::vec, ConstraintType>;
    using typename base_t::constraint_t;

    Param(size_t n, 
          size_t=1,
          constraint_t c = constraint_t())
        : base_t(&i_pack_, n, 1, c)
    {}

private:
    details::ParamInfoPack i_pack_;
};

template <class ValueType, class ConstraintType>
struct Param<ValueType, ppl::mat, ConstraintType> : 
    ParamView<ValueType, ppl::mat, ConstraintType>,
    util::ParamBase<Param<ValueType, ppl::mat, ConstraintType>>
{
    using base_t = ParamView<ValueType, ppl::mat, ConstraintType>;
    using typename base_t::constraint_t;

    Param(size_t rows, size_t cols, constraint_t c = constraint_t())
        : base_t(&i_pack_, rows, cols, c)
    {}

private:
    details::ParamInfoPack i_pack_;
};


// Helper function to create a Param object and deduce the constraint expression.

template <class ValueType
        , class ShapeType = scl
        , class ConstraintType = expr::constraint::Unconstrained>
constexpr inline auto make_param(size_t rows,
                                 size_t cols,
                                 const ConstraintType& c = ConstraintType())
{
    return Param<ValueType, ShapeType, ConstraintType>(rows, cols, c);
}

template <class ValueType
        , class ShapeType = scl
        , class ConstraintType = expr::constraint::Unconstrained>
constexpr inline auto make_param(size_t rows,
                                 const ConstraintType& c = ConstraintType())
{
    return Param<ValueType, ShapeType, ConstraintType>(rows, 1, c);
}

template <class ValueType
        , class ShapeType = scl
        , class ConstraintType = expr::constraint::Unconstrained>
constexpr inline auto make_param(const ConstraintType& c = ConstraintType())
{
    return Param<ValueType, ShapeType, ConstraintType>(1, 1, c);
}

} // namespace ppl

#undef PPL_PARAMVIEW_SHAPE_UNSUPPORTED
#undef PPL_PARAM_SHAPE_UNSUPPORTED
