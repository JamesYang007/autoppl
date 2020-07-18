#pragma once
#include <autoppl/util/traits/type_traits.hpp>
#if __cplusplus <= 201703L
#include <autoppl/util/traits/concept.hpp>
#else
#include <concepts>
#endif
#include <cstddef>

namespace ppl {

inline constexpr size_t DIM_SCALAR = 0;
inline constexpr size_t DIM_VECTOR = 1;
inline constexpr size_t DIM_MATRIX = 2;

/**
 * Class tags to determine which shape a Data or Param is expected to be.
 */
struct scl { static constexpr size_t dim = DIM_SCALAR; };
struct vec { static constexpr size_t dim = DIM_VECTOR; };
struct mat { static constexpr size_t dim = DIM_MATRIX; };

namespace util {

template <class T>
struct shape_traits
{
    using shape_t = typename T::shape_t;
};

#if __cplusplus <= 201703L

/**
 * C++17 version of concepts to check var properties.
 * - var_traits must be well-defined under type T
 * - T must be explicitly convertible to its value_t
 * - not possible to get overloads
 */

template <class T>
inline constexpr bool is_scl_v =
    has_type_shape_t_v<T> &&
    std::is_same_v<get_type_shape_t_t<T>, ppl::scl> &&
    has_func_size_v<const T>
    ;
DEFINE_ASSERT_ONE_PARAM(is_scl_v);

template <class T>
inline constexpr bool is_vec_v =
    has_type_shape_t_v<T> &&
    std::is_same_v<get_type_shape_t_t<T>, ppl::vec> &&
    has_func_size_v<const T>
    ;
DEFINE_ASSERT_ONE_PARAM(is_vec_v);

template <class T>
inline constexpr bool is_mat_v =
    has_type_shape_t_v<T> &&
    std::is_same_v<get_type_shape_t_t<T>, ppl::mat> &&
    has_func_size_v<const T>
    ;
DEFINE_ASSERT_ONE_PARAM(is_mat_v);

template <class T>
inline constexpr bool is_shape_v =
    is_scl_v<T> ||
    is_vec_v<T> ||
    is_mat_v<T>
    ;
DEFINE_ASSERT_ONE_PARAM(is_shape_v);

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

namespace details {

template <class S1
        , class S2
        , bool=is_shape_tag_v<S1> && is_shape_tag_v<S2>>
struct max_shape;

template <class S1, class S2>
struct max_shape<S1, S2, true>
{
    using type = std::conditional_t<
        S1::dim >= S2::dim,
        S1,
        S2>;
};

} // namespace details

/**
 * Returns the type whose shape has more dimension.
 * Undefined behavior if S1 and S2 are not shape tags.
 */
template <class S1, class S2>
using max_shape_t = typename details::max_shape<S1, S2>::type;

} // namespace util
} // namespace ppl
