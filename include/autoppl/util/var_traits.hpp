#pragma once
#include <autoppl/util/type_traits.hpp>
#include <autoppl/util/concept.hpp>

namespace ppl {
namespace util {

/*
 * Traits for Variable-like classes.
 * value_t      type of value Variable represents during computation
 * pointer_t    storage pointer type 
 * state_t      type of enum class state; must have "data" and "parameter"
 */
template <class VarType>
struct var_traits
{
    using value_t = typename VarType::value_t;
    using pointer_t = typename VarType::pointer_t;
    using state_t = typename VarType::state_t;
};

/*
 * C++17 version of concepts to check var properties.
 * - var_traits must be well-defined under type T
 * - T must be explicitly convertible to its value_t
 * - not possible to get overloads
 */
template <class T>
inline constexpr bool is_var_v = 
    has_type_value_t_v<T> &&
    has_type_pointer_t_v<T> &&
    has_type_const_pointer_t_v<T> &&
    has_type_state_t_v<T> &&
    has_func_set_value_v<T> &&
    has_func_get_value_v<const T> &&
    has_func_set_storage_v<T> &&
    has_func_set_state_v<T> &&
    has_func_get_state_v<const T> &&
    is_explicitly_convertible_v<const T, get_type_value_t_t<T>>
    ;

} // namespace util
} // namespace ppl
