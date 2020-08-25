#pragma once
#include <cstdint>
#include <autoppl/util/traits/type_traits.hpp>
#include <autoppl/util/traits/var_expr_traits.hpp>
#include <autoppl/util/traits/concept.hpp>

namespace ppl {
namespace util {

template <class T>
struct VarBase : BaseCRTP<T>
{ using BaseCRTP<T>::self; };

template <class T>
struct ParamBase : VarBase<T>
{};

template <class T>
struct TParamBase : VarBase<T>
{};

template <class T>
struct DataBase : VarBase<T>
{};

template <class T>
inline constexpr bool param_is_base_of_v =
    std::is_base_of_v<ParamBase<T>, T>;

template <class T>
inline constexpr bool tparam_is_base_of_v =
    std::is_base_of_v<TParamBase<T>, T>;

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
    static constexpr bool is_cont_v = util::is_cont_v<typename base_t::value_t>;
    static constexpr bool is_disc_v = util::is_disc_v<typename base_t::value_t>;

    static_assert(is_cont_v == !is_disc_v,
                  PPL_CONT_XOR_DISC); 
};

template <class VarType>
struct param_traits : var_traits<VarType>
{
    using constraint_t = typename VarType::constraint_t;
};

template <class VarType>
struct tparam_traits : var_traits<VarType>
{};

template <class VarType>
struct data_traits : var_traits<VarType>
{};

#if __cplusplus <= 201703L

template <class T>
inline constexpr bool is_param_v = 
    param_is_base_of_v<T> &&
    has_type_id_t_v<T> &&
    has_func_id_v<const T>
    ;

template <class T>
inline constexpr bool is_tparam_v = 
    tparam_is_base_of_v<T> &&
    has_type_id_t_v<T> &&
    has_func_id_v<const T>
    ;

template <class T>
inline constexpr bool is_data_v = 
    data_is_base_of_v<T> &&
    has_type_id_t_v<T> &&
    has_func_id_v<const T>
    ;

// A variable is one that can be assigned a distribution,
// i.e. a random variable.
template <class T>
inline constexpr bool is_var_v = 
    is_param_v<T> || 
    is_tparam_v<T> ||
    is_data_v<T>
    ;

template <class T>
inline constexpr bool is_dist_assignable_v =
    is_param_v<T> ||
    is_data_v<T>
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
    } &&
    requires (T x, const T cx, size_t i) {
        { cx.id() } -> std::same_as<typename var_traits<T>::id_t>;
    }
    ;

template <class T>
concept tparam_c = 
    var_expr_c<T> &&
    tparam_is_base_of_v<T> &&
    requires () {
        typename var_traits<T>::id_t;
    } &&
    requires (T x, const T cx, size_t i) {
        { cx.id() } -> std::same_as<typename var_traits<T>::id_t>;
    }
    ;

template <class T>
concept var_c = 
    param_c<T> ||
    tparam_c<T> ||
    data_c<T>
    ;   

template <class T>
concept dist_assignable_c =
    param_c<T> ||
    data_c<T>
    ;

template <class T>
concept is_data_v = data_c<T>;
template <class T>
concept is_param_v = param_c<T>;
template <class T>
concept is_tparam_v = tparam_c<T>;
template <class T>
concept is_var_v = var_c<T>;
template <class T>
concept is_dist_assignable_v = dist_assignable_c<T>;

#endif

} // namespace util
} // namespace ppl
