#pragma once
#include <functional>
#include <fastad_bits/node.hpp>
#include <armadillo>
#include <autoppl/util/traits/var_traits.hpp>
#include <autoppl/util/traits/shape_traits.hpp>
#include <autoppl/util/traits/var_expr_traits.hpp>
#include <autoppl/util/functional.hpp>

#define PPL_DATA_SHAPE_UNSUPPORTED \
    "Unsupported shape for Data. "
#define PPL_DATAVIEW_SHAPE_UNSUPPORTED \
    "Unsupported shape for DataView. "

namespace ppl {
namespace details {

/**
 * Helper metatool to get underlying value type of a matrix.
 * Specialized for armadillo matrix types.
 * Otherwise, assume the object has member alias value_type.
 */
template <class MatType>
struct mat_value_type
{
    using type = typename MatType::value_type;
};

template <class T>
struct mat_value_type<arma::Mat<T>>
{
    using type = T;
};

template <class T>
using mat_value_type_t = typename
    mat_value_type<T>::type;

} // namespace details

/**
 * DataView is a class that only views data values.
 * It cannot modify the underlying value.
 * If there are multiple values, i.e. shape is vec or mat,
 * it views all of the elements.
 * Specializations for ppl::scl, vec, and mat are provided 
 * and all else are disabled.
 *
 * @tparam  ValueType   underlying value type (usually double or int).
 * @tparam  ShapeType   one of the three shape tags.
 */
template <class ValueType
        , class ShapeType = ppl::scl>
struct DataView
{
    static_assert(util::is_shape_v<ShapeType>,
                  PPL_DATAVIEW_SHAPE_UNSUPPORTED);
};

template <class ValueType>
struct DataView<ValueType, ppl::scl>:
    util::VarExprBase<DataView<ValueType, ppl::scl>>,
    util::DataBase<DataView<ValueType, ppl::scl>>
{
    using value_t = ValueType;
    using const_pointer_t = const value_t*;
    using id_t = const void*;
    using shape_t = ppl::scl;
    static constexpr bool has_param = false;
    static constexpr size_t fixed_size = 1;
    using index_t = uint32_t;

    DataView(const value_t& v) noexcept
        : value_ptr_{&v} 
        , id_{this}
    {}

    template <class VecType
            , class F = util::identity>
    value_t value(const VecType&,
                  size_t=0,
                  F = F()) const 
    { return *value_ptr_; }

    constexpr size_t size() const { return fixed_size; }
    id_t id() const { return id_; }

    template <class VecADVarType>
    auto to_ad(const VecADVarType&,
               const VecADVarType&,
               size_t=0) const 
    { return ad::constant(*value_ptr_); }

    index_t set_cache_offset(index_t idx) const 
    { return idx; }

private:
    const_pointer_t value_ptr_;
    id_t id_;
};

template <class VecType>
struct DataView<VecType, ppl::vec> :
    util::VarExprBase<DataView<VecType, ppl::vec>>,
    util::DataBase<DataView<VecType, ppl::vec>>
{
    using vec_t = VecType;
    using vec_const_pointer_t = const vec_t*;
    using value_t = typename vec_t::value_type;
    using id_t = const void*;
    using shape_t = ppl::vec;
    using index_t = uint32_t;
    static constexpr bool has_param = false;
    static constexpr size_t fixed_size = 0;

    DataView(const vec_t& v) noexcept
        : vec_ptr_{&v}
        , id_{this}
    {}

    template <class PVecType
            , class F = util::identity>
    value_t value(const PVecType&,
                  size_t i,
                  F = F()) const 
    { return (*vec_ptr_)[i]; }

    size_t size() const { return vec_ptr_->size(); }

    id_t id() const { return id_; }

    template <class VecADVarType>
    auto to_ad(const VecADVarType&,
               const VecADVarType&,
               size_t i) const 
    { return ad::constant((*vec_ptr_)[i]); }

    index_t set_cache_offset(index_t idx) const 
    { return idx; }

private:
    vec_const_pointer_t vec_ptr_;
    id_t id_;
};

template <class MatType>
struct DataView<MatType, ppl::mat> :
    util::VarExprBase<DataView<MatType, ppl::mat>>,
    util::DataBase<DataView<MatType, ppl::mat>>
{
    using mat_t = MatType;
    using mat_const_pointer_t = const mat_t*;
    using value_t = details::mat_value_type_t<MatType>;
    using id_t = const void*;
    using shape_t = ppl::mat;
    using index_t = uint32_t;
    static constexpr bool has_param = false;
    static constexpr size_t fixed_size = 0;

    DataView(const mat_t& m) noexcept
        : mat_ptr_{&m}
        , id_{this}
    {}

    template <class PVecType
            , class F = util::identity>
    value_t value(const PVecType&,
                  size_t i,
                  size_t j,
                  F = F()) const 
    { return (*mat_ptr_)(i,j); }

    size_t size() const { return mat_ptr_->n_elem; }
    size_t nrows() const { return mat_ptr_->n_rows; }
    size_t ncols() const { return mat_ptr_->n_cols; }

    id_t id() const { return id_; }

    template <class VecADVarType>
    auto to_ad(const VecADVarType&,
               const VecADVarType&,
               size_t i,
               size_t j) const 
    { return ad::constant((*mat_ptr_)(i,j)); }

    index_t set_cache_offset(index_t idx) const 
    { return idx; }

private:
    mat_const_pointer_t mat_ptr_;
    id_t id_;
};

/**
 * Data a user-friendly wrapper of DataView.
 * It is a DataView (it views itself).
 * The difference is that it owns a container of values.
 * This will usually be used as a quick means to add
 * values directly into a data object. 
 * Otherwise, using DataView through the helper function ppl::make_data_view.
 *
 * @tparam  ValueType   underlying value type (usually double or int)
 * @tparam  ShapeType   one of the three shape tags.
 *                      Currently ppl::mat is not supported.
 *                      Note that it is supported for DataView.
 */

template <class ValueType
        , class ShapeType = ppl::scl>
struct Data
{
    static_assert(util::is_scl_v<ShapeType> ||
                  util::is_vec_v<ShapeType>,
                  PPL_DATA_SHAPE_UNSUPPORTED);
};

// Specialization: scalar
template <class ValueType>
struct Data<ValueType, ppl::scl>:
    DataView<ValueType, ppl::scl>,
    util::VarExprBase<Data<ValueType, ppl::scl>>,
    util::DataBase<Data<ValueType, ppl::scl>>
{
    using base_t = DataView<ValueType, ppl::scl>;
    using typename base_t::value_t;
    using typename base_t::shape_t;
    using typename base_t::id_t;
    using base_t::value;
    using base_t::size;
    using base_t::id;
    using base_t::to_ad;

    Data(value_t v) noexcept
        : base_t(value_)
        , value_(v)
    {}
    Data() noexcept : Data(0) {}

private:
    value_t value_;  // store value associated with data
};

// Specialization: vector
template <class ValueType>
struct Data<ValueType, ppl::vec>: 
    DataView<std::vector<ValueType>, ppl::vec>,
    util::VarExprBase<Data<ValueType, ppl::vec>>,
    util::DataBase<Data<ValueType, ppl::vec>>
{
    using base_t = DataView<std::vector<ValueType>, ppl::vec>;
    using typename base_t::value_t;
    using typename base_t::shape_t;
    using typename base_t::id_t;
    using base_t::value;
    using base_t::size;
    using base_t::id;
    using base_t::to_ad;

    Data(std::initializer_list<value_t> l) noexcept
        : base_t(vec_)
        , vec_(l)
    {}

    Data(size_t n)
        : base_t(vec_)
        , vec_(n)
    {}

    Data() noexcept : Data(0) {}

    void push_back(value_t x) { vec_.push_back(x); }

private:
    std::vector<value_t> vec_;
};

// TODO: Specialization: mat-like

template <class ShapeType = ppl::scl, class Container>
inline constexpr auto make_data_view(const Container& x) 
{ return DataView<Container, ShapeType>(x); }

} // namespace ppl

#undef PPL_DATA_SHAPE_UNSUPPORTED
#undef PPL_DATAVIEW_SHAPE_UNSUPPORTED
