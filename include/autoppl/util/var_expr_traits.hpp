#pragma once
#include <autoppl/util/type_traits.hpp>
#include <autoppl/util/concept.hpp>
#include <autoppl/util/var_traits.hpp>

namespace ppl {
namespace util {

/*
 * Traits for Variable Expression classes.
 * value_t      type of value Variable represents during computation
 */
template <class VarExprType>
struct var_expr_traits
{
    using value_t = typename VarExprType::value_t;
};

// Specialization: when double or int, considered "trivial" variable.
// TODO: this was a quick fix for generalizing distribution value_t.
template <>
struct var_expr_traits<double>
{
    using value_t = double;
};

/*
 * A variable expression is any class that satisfies the following concept.
 */
template <class T>
inline constexpr bool is_var_expr_v = 
    !is_var_v<T> &&
    has_type_value_t_v<T> &&
    has_func_get_value_v<const T> &&
    is_explicitly_convertible_v<const T, get_type_value_t_t<T>>
    ;

namespace details {

// Tool needed to assert 
template <class T>
inline constexpr bool is_not_var_v = !is_var_v<T>;
DEFINE_ASSERT_ONE_PARAM(is_not_var_v);

} // namespace details 

template <class T>
inline constexpr bool assert_is_var_expr_v = 
    details::assert_is_not_var_v<T> &&
    assert_has_type_value_t_v<T> &&
    assert_has_func_get_value_v<const T> &&
    assert_is_explicitly_convertible_v<const T, get_type_value_t_t<T>>
    ;

} // namespace util
} // namespace ppl
