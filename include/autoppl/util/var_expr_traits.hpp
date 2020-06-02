#pragma once
#include <autoppl/util/type_traits.hpp>
#include <autoppl/util/concept.hpp>
#include <autoppl/util/var_traits.hpp>

namespace ppl {
namespace util {

/**
 * Base class for all variable expressions.
 * It is necessary for all variable expressions to
 * derive from this class.
 */
template <class T>
struct VarExpr : BaseCRTP<T>
{ using BaseCRTP<T>::self; };

/**
 * Checks if VarExpr<T> is base of type T 
 */
template <class T>
inline constexpr bool var_expr_is_base_of_v =
    std::is_base_of_v<VarExpr<T>, T>;

DEFINE_ASSERT_ONE_PARAM(var_expr_is_base_of_v);

/**
 * Traits for Variable Expression classes.
 * value_t      type of value Variable represents during computation
 */
template <class VarExprType>
struct var_expr_traits
{
    using value_t = typename VarExprType::value_t;
};

/**
 * A variable expression is any class that satisfies the following concept.
 */
template <class T>
inline constexpr bool is_var_expr_v = 
    var_expr_is_base_of_v<T> &&
    !is_var_v<T> &&
    has_type_value_t_v<T> &&
    has_func_get_value_v<const T>
    ;

namespace details {

// Tool needed to assert 
template <class T>
inline constexpr bool is_not_var_v = !is_var_v<T>;
DEFINE_ASSERT_ONE_PARAM(is_not_var_v);

} // namespace details 

template <class T>
inline constexpr bool assert_is_var_expr_v = 
    assert_var_expr_is_base_of_v<T> &&
    details::assert_is_not_var_v<T> &&
    assert_has_type_value_t_v<T> &&
    assert_has_func_get_value_v<const T>
    ;

} // namespace util
} // namespace ppl
