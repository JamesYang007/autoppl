#pragma once
#if __cplusplus <= 201703L
#include <autoppl/util/traits/concept.hpp>
#else
#include <concepts>
#endif
#include <autoppl/util/traits/type_traits.hpp>
#include <autoppl/util/traits/shape_traits.hpp>

namespace ppl {
namespace util {

/**
 * Base class for all variable expressions.
 * It is necessary for all variable expressions to
 * derive from this class.
 */
template <class T>
struct VarExprBase : BaseCRTP<T>
{ using BaseCRTP<T>::self; };

template <class T>
inline constexpr bool var_expr_is_base_of_v =
    std::is_base_of_v<VarExprBase<T>, T>;

template <class VarExprType>
struct var_expr_traits
{
    using value_t = typename VarExprType::value_t;
    static constexpr bool has_param = VarExprType::has_param;
};

#if __cplusplus <= 201703L

template <class T>
inline constexpr bool is_var_expr_v = 
    is_shape_v<T> &&
    var_expr_is_base_of_v<T> &&
    has_type_value_t_v<T>
    ;

#else

template <class T>
concept var_expr_c = 
    shape_c<T> &&
    var_expr_is_base_of_v<T> &&
    requires () {
        var_expr_traits<T>::has_param;
        typename var_expr_traits<T>::value_t;
    } //&&
    //(
    //    ( 
    //        !util::is_mat_v<T> &&
    //        requires (const MockVector<typename var_expr_traits<T>::value_t>& values,
    //                  const MockVector< ad::Var<
    //                    typename var_expr_traits<T>::value_t> >& ad_vars,
    //                  const T& cx, 
    //                  size_t i) {
    //            { cx.value(values, i) } -> std::convertible_to<
    //                typename var_expr_traits<T>::value_t>;
    //            { cx.to_ad(ad_vars, ad_vars, i) } -> ad::is_ad_expr;
    //        }
    //    ) ||       
    //    ( 
    //        util::is_mat_v<T> &&
    //        requires (const MockVector<typename var_expr_traits<T>::value_t>& values,
    //                  const MockVector< ad::Var<
    //                    typename var_expr_traits<T>::value_t> >& ad_vars,
    //                  const T& cx, 
    //                  size_t i) {
    //            { cx.value(values, i, i) } -> std::convertible_to<
    //                typename var_expr_traits<T>::value_t>;
    //            { cx.to_ad(ad_vars, ad_vars, i, i) } -> ad::is_ad_expr;
    //        }
    //    ) 
    //)
    ;

template <class T>
concept is_var_expr_v = var_expr_c<T>;

#endif


} // namespace util
} // namespace ppl
