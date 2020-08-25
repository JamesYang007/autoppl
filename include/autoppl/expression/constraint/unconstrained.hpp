#pragma once
#include <fastad_bits/util/shape_traits.hpp>
#include <fastad_bits/reverse/core/var_view.hpp>
#include <autoppl/expression/constraint/transformer.hpp>
#include <autoppl/util/value.hpp>

namespace ppl {
namespace expr {
namespace constraint {

struct Unconstrained {};

template <class ValueType
        , class ShapeType>
struct Transformer<ValueType, ShapeType, Unconstrained>
{
    using value_t = ValueType;
    using shape_t = ShapeType;
    using constraint_t = Unconstrained;
    using var_t = util::var_t<value_t, shape_t>;
    using view_t = ad::util::shape_to_raw_view_t<value_t, shape_t>;

    /**
     * Constructs a Transformer object.
     *
     * @param   rows    number of constrained rows
     * @param   cols    number of constrained cols
     */
    Transformer(size_t rows, 
                size_t cols, 
                constraint_t=constraint_t())
        : uc_val_(util::make_val<value_t, shape_t>(rows, cols))
    {
        // TODO: remove?
        //if constexpr (!util::is_scl_v<std::decay_t<decltype(*this)>>) {
        //    util::bind(uc_val_, nullptr, rows, cols); 
        //}
    }

    void transform() {}
    void inv_transform(size_t) {}

    template <class CurrPtrPack, class PtrPack>
    auto inv_transform_ad(const CurrPtrPack& curr_pack,
                          const PtrPack&,
                          size_t) const {
        return ad::VarView<value_t, shape_t>(
                curr_pack.uc_val, curr_pack.uc_adj, rows_uc(), cols_uc());
    }

    template <class CurrPtrPack, class PtrPack>
    auto logj_inv_transform_ad(const CurrPtrPack&,
                               const PtrPack&) const {
        return ad::constant(0.);
    }

    template <class GenType, class ContDist>
    void init(GenType& gen, ContDist& dist) {
        static_cast<void>(gen);
        static_cast<void>(dist);
        if constexpr (util::is_disc_v<value_t>) {
            if constexpr (std::is_same_v<shape_t, scl>) {
                *uc_val_ = 0;
            } else {
                uc_val_.setZero();
            }
        } else {
            if constexpr (std::is_same_v<shape_t, scl>) {
                *uc_val_ = dist(gen);    
            } else {
                uc_val_ = var_t::NullaryExpr(rows_uc(), cols_uc(), [&]() { return dist(gen); });
            }
        }
    }

    void activate_refcnt(size_t) const {}

    var_t& get_c() { return util::get(uc_val_); }
    const var_t& get_c() const { return util::get(uc_val_); }

    constexpr size_t size_uc() const { return util::size(uc_val_); }
    constexpr size_t rows_uc() const { return util::rows(uc_val_); }
    constexpr size_t cols_uc() const { return util::cols(uc_val_); }
    constexpr size_t size_c() const { return size_uc(); }
    constexpr size_t rows_c() const { return rows_uc(); }
    constexpr size_t cols_c() const { return cols_uc(); }

    constexpr size_t bind_size_uc() const { return size_uc(); }
    constexpr size_t bind_size_c() const { return 0; }
    constexpr size_t bind_size_v() const { return 0; }

    /**
     * Binds unconstrained viewer to unconstrained region.
     */
    template <class CurrPtrPack, class PtrPack>
    void bind(const CurrPtrPack& curr_pack,
              const PtrPack&) 
    { util::bind(uc_val_, curr_pack.uc_val, rows_uc(), cols_uc()); }

private:
    view_t uc_val_;
};

} // namespace constraint
} // namespace expr
} // namespace ppl
