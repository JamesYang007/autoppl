#pragma once
#include <fastad_bits/reverse/core/constant.hpp>
#include <autoppl/util/traits/var_expr_traits.hpp>

#define PPL_CONSTANT_SHAPE_UNSUPPORTED \
    "Unsupported shape for constants. "

namespace ppl {
namespace expr {
namespace var {

template <class ValueType
        , class ShapeType=ppl::scl>
struct Constant
{
    static_assert(!util::is_shape_v<ShapeType>,
                  PPL_CONSTANT_SHAPE_UNSUPPORTED);
};

template <class ValueType>
struct Constant<ValueType, ppl::scl>:
    util::VarExprBase<Constant<ValueType, ppl::scl>>
{
    using value_t = ValueType;
    using shape_t = ppl::scl;
    static constexpr bool has_param = false;

    Constant(value_t c) : c_{c} {}

    template <class Func>
    void traverse(Func&&) const {}

    value_t eval() const { return c_; }
    value_t get() const { return c_; }
    constexpr size_t size() const { return 1; }

    template <class PtrPackType>
    auto ad(const PtrPackType&) const
    { return ad::constant(c_); }

    void activate_refcnt() const {}

private:
    value_t c_;
};

template <class ValueType>
struct Constant<ValueType, ppl::vec>:
    util::VarExprBase<Constant<ValueType, ppl::vec>>
{
    using value_t = ValueType;
    using shape_t = ppl::vec;
    static constexpr bool has_param = false;

    template <class T>
    Constant(const Eigen::EigenBase<T>& c) : c_{c} {}

    template <class Func>
    void traverse(Func&&) const {}

    const auto& eval() const { return c_; }
    const auto& get() const { return c_; }
    size_t size() const { return c_.size(); }
    size_t rows() const { return c_.rows(); }
    constexpr size_t cols() const { return 1; }

    template <class PtrPackType>
    auto ad(const PtrPackType&) const
    { return ad::constant_view(c_.data(), rows()); }

    void activate_refcnt() const {}

private:
    Eigen::Matrix<value_t, Eigen::Dynamic, 1> c_;
};

template <class ValueType>
struct Constant<ValueType, ppl::mat>:
    util::VarExprBase<Constant<ValueType, ppl::mat>>
{
    using value_t = ValueType;
    using shape_t = ppl::mat;
    static constexpr bool has_param = false;

    template <class T>
    Constant(const Eigen::EigenBase<T>& c) : c_{c} {}

    template <class Func>
    void traverse(Func&&) const {}

    const auto& eval() const { return c_; }
    const auto& get() const { return c_; }
    size_t size() const { return c_.size(); }
    size_t rows() const { return c_.rows(); }
    size_t cols() const { return c_.cols(); }

    template <class PtrPackType>
    auto ad(const PtrPackType&) const
    { return ad::constant_view(c_.data(), rows(), cols()); }

    void activate_refcnt() const {}

private:
    Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic> c_;
};

} // namespace var
} // namespace expr
} // namespace ppl

#undef PPL_CONSTANT_VEC_UNSUPPORTED 
#undef PPL_CONSTANT_MAT_UNSUPPORTED 
