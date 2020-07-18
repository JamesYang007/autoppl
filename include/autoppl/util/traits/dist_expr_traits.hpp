#pragma once
#if __cplusplus <= 201703L
#include <autoppl/util/traits/concept.hpp>
#endif
#include <autoppl/util/traits/type_traits.hpp>
#include <autoppl/util/traits/var_traits.hpp>
#include <cstdint>
#include <cstddef>

namespace ppl {
namespace util {

/**
 * Base class for all distribution expressions.
 * It is necessary for all distribution expressions to
 * derive from this class.
 */
template <class T>
struct DistExprBase : BaseCRTP<T>
{ 
    using BaseCRTP<T>::self;
    using dist_value_t = double;
};

template <class T>
inline constexpr bool dist_expr_is_base_of_v =
    std::is_base_of_v<DistExprBase<T>, T>;

/**
 * Continuous distribution expressions can be constructed with this type.
 */
using cont_param_t = double;

/**
 * Discrete distribution expressions can be constructed with this type.
 */
using disc_param_t = int32_t;

/**
 * Traits for Distribution Expression classes.
 * value_t      type of value Variable represents during computation
 * dist_value_t type of pdf/log_pdf value
 */
template <class DistExprType>
struct dist_expr_traits
{
    using value_t = typename DistExprType::value_t;
    using dist_value_t = typename DistExprType::dist_value_t;
    using index_t = typename DistExprType::index_t;
    static constexpr bool is_cont_v = util::is_cont_v<value_t>; 
    static constexpr bool is_disc_v = util::is_disc_v<value_t>; 

    static_assert(is_cont_v == !is_disc_v,
                  PPL_CONT_XOR_DISC); 
};

#if __cplusplus <= 201703L

DEFINE_ASSERT_ONE_PARAM(dist_expr_is_base_of_v);

/**
 * A distribution expression is any class that satisfies the following concept:
 */
template <class T>
inline constexpr bool is_dist_expr_v = 
    dist_expr_is_base_of_v<T> &&
    has_type_value_t_v<T> &&
    has_type_dist_value_t_v<T> &&
    has_type_index_t_v<T>
    ;

template <class T>
inline constexpr bool assert_is_dist_expr_v = 
    assert_dist_expr_is_base_of_v<T> &&
    assert_has_type_value_t_v<T> &&
    assert_has_type_dist_value_t_v<T> &&
    assert_has_type_index_t_v<T>
    ;

#else
} // namespace util

// Forward declaration
template <class ValueType
        , class ShapeType>
struct Param;

namespace util {

template <class T>
concept dist_expr_c = 
    dist_expr_is_base_of_v<T> &&
    requires () {
        typename dist_expr_traits<T>::value_t;
        typename dist_expr_traits<T>::dist_value_t;
        typename dist_expr_traits<T>::index_t;
    } &&
    requires(typename var_expr_traits<T>::index_t offset,
             T& x) {
       { x.set_cache_offset(offset) } -> std::same_as<
               typename dist_expr_traits<T>::index_t 
               >;
    } &&
    (
        requires (const ppl::Param<typename dist_expr_traits<T>::value_t, ppl::scl>& p,
                  const MockVector<typename dist_expr_traits<T>::value_t>& v,
                  const T& cx,
                  size_t i) {
            { cx.pdf(p, v) } -> std::same_as<typename dist_expr_traits<T>::dist_value_t>;
            { cx.log_pdf(p, v) } -> std::same_as<typename dist_expr_traits<T>::dist_value_t>;
            { cx.min(v, i) } -> std::same_as<typename dist_expr_traits<T>::value_t>;
            { cx.max(v, i) } -> std::same_as<typename dist_expr_traits<T>::value_t>;
        } ||
        requires (const ppl::Param<typename dist_expr_traits<T>::value_t, ppl::vec>& p,
                  const MockVector<typename dist_expr_traits<T>::value_t>& v,
                  const T& cx,
                  size_t i) {
            { cx.pdf(p, v) } -> std::same_as<typename dist_expr_traits<T>::dist_value_t>;
            { cx.log_pdf(p, v) } -> std::same_as<typename dist_expr_traits<T>::dist_value_t>;
            { cx.min(v, i) } -> std::same_as<typename dist_expr_traits<T>::value_t>;
            { cx.max(v, i) } -> std::same_as<typename dist_expr_traits<T>::value_t>;
        }
    )
    ;

template <class T>
concept is_dist_expr_v = dist_expr_c<T>;

#endif

} // namespace util
} // namespace ppl
