#pragma once
#include <autoppl/util/traits/type_traits.hpp>
#include <fastad_bits/util/shape_traits.hpp>
#if __cplusplus <= 201703L
#include <autoppl/util/traits/concept.hpp>
#else
#include <concepts>
#endif
#include <cstddef>

namespace ppl {

/**
 * Class tags to determine which shape a Data or Param is expected to be.
 */
using scl = ad::scl;
using vec = ad::vec;
using mat = ad::mat;

namespace util {

template <class T>
using shape_traits = ad::util::shape_traits<T>;

#if __cplusplus <= 201703L

template <class T>
inline constexpr bool is_scl_v = ad::util::is_scl_v<T>;

template <class T>
inline constexpr bool is_vec_v = ad::util::is_vec_v<T>;

template <class T>
inline constexpr bool is_mat_v = ad::util::is_mat_v<T>;

template <class T>
inline constexpr bool is_shape_v =
    is_scl_v<T> ||
    is_vec_v<T> ||
    is_mat_v<T> 
    ;

#else

template <class T>
concept scl_c = 
    requires(const T cx) {
        typename T::shape_t;
        { cx.size() } -> std::same_as<size_t>;
    } &&
    std::same_as<typename T::shape_t, ppl::scl>
    ;

template <class T>
concept vec_c = 
    requires(const T cx) {
        typename T::shape_t;
        { cx.size() } -> std::same_as<size_t>;
    } &&
    std::same_as<typename T::shape_t, ppl::vec>
    ;

template <class T>
concept mat_c = 
    requires(const T cx) {
        typename T::shape_t;
        { cx.size() } -> std::same_as<size_t>;
    } &&
    std::same_as<typename T::shape_t, ppl::mat>
    ;

template <class T>
concept shape_c =
    scl_c<T> || 
    vec_c<T> ||
    mat_c<T>
    ;

template <class T>
concept is_scl_v = scl_c<T>;

template <class T>
concept is_vec_v = vec_c<T>;

template <class T>
concept is_mat_v = mat_c<T>;

template <class T>
concept is_shape_v = shape_c<T>;

#endif

//////////////////////////////////////////////////
// Useful tools to manage shapes
//////////////////////////////////////////////////

/**
 * Checks if T is a shape tag.
 */
template <class T>
inline constexpr bool is_shape_tag_v =
    std::is_same_v<T, ppl::scl> ||
    std::is_same_v<T, ppl::vec> ||
    std::is_same_v<T, ppl::mat>
    ;

/**
 * Helper metaprogramming tools for Eigen-related types.
 */
namespace details {

template <class V, class T>
struct var;

template <class V>
struct var<V, scl> { using type = V; };

template <class V>
struct var<V, vec> 
{ 
    using type = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, 1>>; 
};

template <class V>
struct var<V, mat> 
{ 
    using type = Eigen::Map<Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic>>; 
};

} // namespace details
template <class V, class T>
using var_t = typename details::var<V, T>::type;

} // namespace util
} // namespace ppl
