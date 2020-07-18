#pragma once
#include <cstdint>
#include <autoppl/util/traits/type_traits.hpp>
#include <autoppl/util/traits/var_expr_traits.hpp>
#include <autoppl/util/traits/concept.hpp>
#include <autoppl/util/functional.hpp>

/*
 * We say Param or Data, etc. are vars.
 */

namespace ppl {
namespace util {

template <class T>
struct ParamBase : BaseCRTP<T>
{ using BaseCRTP<T>::self; };

template <class T>
struct DataBase : BaseCRTP<T>
{ using BaseCRTP<T>::self; };

template <class T>
inline constexpr bool param_is_base_of_v =
    std::is_base_of_v<ParamBase<T>, T>;

template <class T>
inline constexpr bool data_is_base_of_v =
    std::is_base_of_v<DataBase<T>, T>;

template <class VarType>
struct var_traits : var_expr_traits<VarType>
{
private:
    using base_t = var_expr_traits<VarType>;
public:
    using id_t = typename VarType::id_t;
    using vec_t = get_type_vec_t_t<VarType>;
    using mat_t = get_type_mat_t_t<VarType>;
    static constexpr bool is_cont_v = util::is_cont_v<typename base_t::value_t>;
    static constexpr bool is_disc_v = util::is_disc_v<typename base_t::value_t>;

    static_assert(is_cont_v == !is_disc_v,
                  PPL_CONT_XOR_DISC); 
};

template <class VarType>
struct param_traits : var_traits<VarType>
{
    using pointer_t = typename VarType::pointer_t;
    using const_pointer_t = typename VarType::const_pointer_t;
};

template <class VarType>
struct data_traits : var_traits<VarType>
{};

#if __cplusplus <= 201703L

DEFINE_ASSERT_ONE_PARAM(param_is_base_of_v);
DEFINE_ASSERT_ONE_PARAM(data_is_base_of_v);

template <class T>
inline constexpr bool is_param_v = 
    // T itself is a parameter-like variable
    is_var_expr_v<T> &&
    param_is_base_of_v<T> &&
    has_type_id_t_v<T> &&
    has_type_pointer_t_v<T> &&
    has_type_const_pointer_t_v<T> &&
    has_func_id_v<const T>
    ;

template <class T>
inline constexpr bool is_data_v = 
    is_var_expr_v<T> &&
    data_is_base_of_v<T> &&
    has_type_id_t_v<T> &&
    has_func_id_v<const T>
    ;

template <class T>
inline constexpr bool is_var_v = 
    is_param_v<T> || 
    is_data_v<T>
    ;
DEFINE_ASSERT_ONE_PARAM(is_var_v);

template <class T>
inline constexpr bool assert_is_param_v = 
    assert_is_var_expr_v<T> &&
    assert_param_is_base_of_v<T> &&
    assert_has_type_pointer_t_v<T> &&
    assert_has_type_const_pointer_t_v<T> &&
    assert_has_type_id_t_v<T> &&
    assert_has_func_id_v<const T>
    ;

template <class T>
inline constexpr bool assert_is_data_v = 
    assert_is_var_expr_v<T> &&
    assert_data_is_base_of_v<T> &&
    assert_has_type_id_t_v<T> &&
    assert_has_func_id_v<const T>
    ;

#else 

template <class T>
concept data_c =
    var_expr_c<T> &&
    data_is_base_of_v<T> &&
    requires (const T cx, size_t i) {
        typename var_traits<T>::id_t;
        { cx.id() } -> std::same_as<typename var_traits<T>::id_t>;
    }
    ;

template <class T>
concept param_c = 
    var_expr_c<T> &&
    param_is_base_of_v<T> &&
    requires () {
        typename var_traits<T>::id_t;
        typename param_traits<T>::pointer_t;
        typename param_traits<T>::const_pointer_t;
    } &&
    requires (T x, const T cx, size_t i,
              typename param_traits<T>::index_t offset) {
        { x.set_offset(offset) } -> std::same_as<
                typename var_traits<T>::index_t 
                >;
        { cx.storage(i) } -> std::convertible_to<typename param_traits<T>::pointer_t>;
        { cx.id() } -> std::same_as<typename var_traits<T>::id_t>;
    }
    ;

template <class T>
concept var_c = 
    data_c<T> ||
    param_c<T>
    ;   

template <class T>
concept is_data_v = data_c<T>;
template <class T>
concept is_param_v = param_c<T>;
template <class T>
concept is_var_v = var_c<T>;

#endif

} // namespace util
} // namespace ppl
