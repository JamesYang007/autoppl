#pragma once
#include <cstddef>
#include <autoppl/util/traits/shape_traits.hpp>

namespace ppl {
namespace util {

template <class ValueType, class ShapeType>
constexpr auto make_val(size_t rows=1, size_t cols=1)
{
    static_cast<void>(rows);
    static_cast<void>(cols);
    if constexpr (std::is_same_v<ShapeType, scl>) {
        return nullptr;
    } else {
        using map_t = ad::util::shape_to_raw_view_t<ValueType, ShapeType>;
        return map_t(nullptr, rows, cols);
    }
}

template <class T>
constexpr size_t size(const T& x)
{
    if constexpr (std::is_pointer_v<std::decay_t<T>>) {
        return 1;
    } else {
        return x.size(); 
    }
}

template <class T>
constexpr size_t rows(const T& x)
{
    if constexpr (std::is_pointer_v<std::decay_t<T>>) {
        return 1;
    } else {
        return x.rows(); 
    }
}

template <class T>
constexpr size_t cols(const T& x)
{
    if constexpr (std::is_pointer_v<std::decay_t<T>>) {
        return 1;
    } else {
        return x.cols(); 
    }
}

template <class T>
auto& get(T&& x) {
    if constexpr (std::is_pointer_v<std::decay_t<T>>) {
        return *x;
    } else {
        return x; 
    }
}

template <class T, class ValPtrType>
void bind(T& x, ValPtrType begin, size_t rows=1, size_t cols=1)
{
    static_cast<void>(rows);
    static_cast<void>(cols);
    using raw_t = std::decay_t<T>;
    if constexpr (std::is_pointer_v<raw_t>) {
        x = begin;
    } else {
        new (&x) raw_t(begin, rows, cols);
    }
}

template <class T>
constexpr inline auto to_array(const T& x) 
{
    using x_t = std::decay_t<decltype(x)>;
    if constexpr (std::is_arithmetic_v<x_t>) return x;
    else return x.array();
}

} // namespace util
} // namespace ppl 
