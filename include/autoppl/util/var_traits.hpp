#pragma once
#include <autoppl/util/type_traits.hpp>
#include <autoppl/util/concept.hpp>

namespace ppl {
namespace util {

/*
 * Base class for all variables.
 * It is necessary for all variables to
 * derive from this class.
 */
template <class T>
struct Var : BaseCRTP<T> 
{ using BaseCRTP<T>::self; };

template <class T>
struct DataLike : Var<T>
{ using Var<T>::self; };

template <class T>
struct ParamLike : Var<T>
{ using Var<T>::self; };


/*
 * Checks if DistExpr<T> is base of type T 
 */

template <class T>
inline constexpr bool data_is_base_of_v =
    std::is_base_of_v<DataLike<T>, T>;

template <class T>
inline constexpr bool param_is_base_of_v =
    std::is_base_of_v<ParamLike<T>, T>;

template <class T>
inline constexpr bool var_is_base_of_v =
    std::is_base_of_v<Var<T>, T>;

DEFINE_ASSERT_ONE_PARAM(var_is_base_of_v);
DEFINE_ASSERT_ONE_PARAM(param_is_base_of_v);
DEFINE_ASSERT_ONE_PARAM(data_is_base_of_v);

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
};

/*
 * C++17 version of concepts to check var properties.
 * - var_traits must be well-defined under type T
 * - T must be explicitly convertible to its value_t
 * - not possible to get overloads
 */
template <class T>
inline constexpr bool is_data_v = 
    data_is_base_of_v<T> &&
    has_type_value_t_v<T> &&
    has_type_pointer_t_v<T> &&
    has_type_const_pointer_t_v<T> &&
    has_func_get_value_v<const T>
    ;

template <class T>
inline constexpr bool is_param_v = 
    param_is_base_of_v<T> &&
    has_type_value_t_v<T> &&
    has_type_pointer_t_v<T> &&
    has_type_const_pointer_t_v<T> &&
    has_func_set_value_v<T> &&
    has_func_get_value_v<const T> &&
    has_func_set_storage_v<T>
    ;

template <class T>
inline constexpr bool assert_is_data_v = 
    assert_data_is_base_of_v<T> &&
    assert_has_type_value_t_v<T> &&
    assert_has_type_pointer_t_v<T> &&
    assert_has_type_const_pointer_t_v<T> &&
    assert_has_func_get_value_v<const T>
    ;

template <class T>
inline constexpr bool assert_is_param_v = 
    assert_param_is_base_of_v<T> &&
    assert_has_type_value_t_v<T> &&
    assert_has_type_pointer_t_v<T> &&
    assert_has_type_const_pointer_t_v<T> &&
    assert_has_func_set_value_v<T> &&
    assert_has_func_get_value_v<const T> &&
    assert_has_func_set_storage_v<T>
    ;

template <class T>
inline constexpr bool is_var_v =
    is_data_v<T> || is_param_v<T>
    ;

DEFINE_ASSERT_ONE_PARAM(is_var_v);

} // namespace util
} // namespace ppl
