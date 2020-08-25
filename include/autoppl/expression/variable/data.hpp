#pragma once
#include <Eigen/Dense>
#include <fastad_bits/reverse/core/constant.hpp>
#include <autoppl/util/traits/var_traits.hpp>
#include <autoppl/util/traits/shape_traits.hpp>
#include <autoppl/util/traits/var_expr_traits.hpp>

#define PPL_DATA_SHAPE_UNSUPPORTED \
    "Unsupported shape for Data. "
#define PPL_DATAVIEW_SHAPE_UNSUPPORTED \
    "Unsupported shape for DataView. "

namespace ppl {

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
    using var_t = value_t;
    using id_t = const void*;
    using shape_t = ppl::scl;
    static constexpr bool has_param = false;

    DataView(const value_t* begin) noexcept
        : var_{begin} 
        , id_{this}
    {}

    template <class Func>
    void traverse(Func&&) const {}

    const var_t& eval() const { return get(); }
    const var_t& get() const { return *var_; }

    constexpr size_t size() const { return 1; }
    constexpr size_t rows() const { return 1; }
    constexpr size_t cols() const { return 1; }
    id_t id() const { return id_; }

    template <class PtrPackType>
    auto ad(const PtrPackType&) const
    { return ad::constant(*var_); }

    template <class PtrType>
    void bind(PtrType begin) 
    { 
        static_cast<void>(begin);
        if constexpr (std::is_convertible_v<PtrType, value_t*>) {
            var_ = begin; 
        }
    }

    void activate_refcnt() const {}

private:
    const var_t* var_;
    id_t id_;
};

template <class ValueType>
struct DataView<ValueType, ppl::vec> :
    util::VarExprBase<DataView<ValueType, ppl::vec>>,
    util::DataBase<DataView<ValueType, ppl::vec>>
{
    using value_t = ValueType;
    using var_t = Eigen::Map<const Eigen::Matrix<value_t, Eigen::Dynamic, 1>>;
    using id_t = const void*;
    using shape_t = ppl::vec;
    static constexpr bool has_param = false;

    DataView(const value_t* begin,
             size_t rows) noexcept
        : var_(begin, rows)
        , id_{this}
    {}

    const var_t& eval() const { return get(); }
    const var_t& get() const { return var_; }
    size_t size() const { return var_.size(); }
    size_t rows() const { return var_.rows(); }
    constexpr size_t cols() const { return 1; }
    id_t id() const { return id_; }

    template <class PtrPackType>
    auto ad(const PtrPackType&) const
    { return ad::constant_view(var_.data(), size()); }

    template <class PtrType>
    void bind(PtrType begin) 
    { 
        static_cast<void>(begin);
        if constexpr (std::is_convertible_v<PtrType, value_t*>) {
            new (&var_) var_t(begin, size()); 
        }
    }

    void activate_refcnt() const {}

private:
    var_t var_;
    id_t id_;
};

template <class ValueType>
struct DataView<ValueType, ppl::mat> :
    util::VarExprBase<DataView<ValueType, ppl::mat>>,
    util::DataBase<DataView<ValueType, ppl::mat>>
{
    using value_t = ValueType;
    using var_t = Eigen::Map<const Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>>;
    using id_t = const void*;
    using shape_t = ppl::mat;
    static constexpr bool has_param = false;

    DataView(const value_t* begin,
             size_t rows,
             size_t cols) noexcept
        : var_(begin, rows, cols)
        , id_{this}
    {}

    const var_t& eval() const { return get(); }
    const var_t& get() const { return var_; }
    size_t size() const { return var_.size(); }
    size_t rows() const { return var_.rows(); }
    size_t cols() const { return var_.cols(); }
    id_t id() const { return id_; }

    template <class PtrPackType>
    auto ad(const PtrPackType&) const
    { return ad::constant_view(var_.data(), rows(), cols()); }

    template <class PtrType>
    void bind(PtrType begin) 
    { 
        static_cast<void>(begin);
        if constexpr (std::is_convertible_v<PtrType, value_t*>) {
            new (&var_) var_t(begin, rows(), cols()); 
        }
    }

    void activate_refcnt() const {}

private:
    var_t var_;
    id_t id_;
};

/**
 * Data a user-friendly wrapper of DataView.
 * It is a DataView (it views itself).
 * The difference is that it owns the container of values.
 * This will usually be used as a quick means to add
 * values directly into a data object. 
 *
 * A Data is not a variable expression, but a DataView is.
 * This means any model expression that references Data objects will
 * not create copies of underlying container.
 *
 * @tparam  ValueType   underlying value type (usually double or int)
 * @tparam  ShapeType   one of the three shape tags.
 */

template <class ValueType
        , class ShapeType = ppl::scl>
struct Data
{
    static_assert(util::is_shape_v<ShapeType>,
                  PPL_DATA_SHAPE_UNSUPPORTED);
};

// Specialization: scalar
template <class ValueType>
struct Data<ValueType, ppl::scl>:
    DataView<ValueType, ppl::scl>,
    util::DataBase<Data<ValueType, ppl::scl>>
{
    using base_t = DataView<ValueType, ppl::scl>;
    using typename base_t::value_t;
    using base_t::get;

    Data(value_t v) noexcept
        : base_t(&value_)
        , value_(v)
    {}
    Data() noexcept : Data(0) {}

    auto& get() { return value_; }

private:
    value_t value_;  // store value associated with data
};

// Specialization: vector
template <class ValueType>
struct Data<ValueType, ppl::vec>: 
    DataView<ValueType, ppl::vec>,
    util::DataBase<Data<ValueType, ppl::vec>>
{
    using base_t = DataView<ValueType, ppl::vec>;
    using typename base_t::value_t;
    using base_t::bind;
    using base_t::get;

    Data(size_t n)
        : base_t(nullptr, n)
        , vec_(n)
    { this->bind(vec_.data()); }

    auto& get() { return vec_; }

private:
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    vec_t vec_;
};

// Specialization: matrix
template <class ValueType>
struct Data<ValueType, ppl::mat>: 
    DataView<ValueType, ppl::mat>,
    util::DataBase<Data<ValueType, ppl::mat>>
{
    using base_t = DataView<ValueType, ppl::mat>;
    using typename base_t::value_t;
    using base_t::bind;
    using base_t::get;

    Data(size_t rows, size_t cols)
        : base_t(nullptr, rows, cols)
        , mat_(rows, cols)
    { this->bind(mat_.data()); }

    auto& get() { return mat_; }

private:
    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;
    mat_t mat_;
};

} // namespace ppl

#undef PPL_DATA_SHAPE_UNSUPPORTED
#undef PPL_DATAVIEW_SHAPE_UNSUPPORTED
