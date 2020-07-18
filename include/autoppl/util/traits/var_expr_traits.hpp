#pragma once
#if __cplusplus <= 201703L
#include <autoppl/util/traits/concept.hpp>
#else
#include <concepts>
#endif
#include <autoppl/util/traits/type_traits.hpp>
#include <autoppl/util/traits/shape_traits.hpp>
#include <autoppl/util/ad_boost/type_traits.hpp>

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
    using index_t = typename VarExprType::index_t;
    static constexpr bool has_param = VarExprType::has_param;
    static constexpr size_t fixed_size = VarExprType::fixed_size;
};

template <class VarExprType>
inline constexpr bool is_fixed_size_v =
    var_expr_traits<VarExprType>::fixed_size > 0;

#if __cplusplus <= 201703L

DEFINE_ASSERT_ONE_PARAM(var_expr_is_base_of_v);

/**
 * A variable expression is any class that satisfies the following concept.
 */
template <class T>
inline constexpr bool is_var_expr_v = 
    is_shape_v<T> &&
    var_expr_is_base_of_v<T> &&
    has_type_value_t_v<T> &&
    has_type_index_t_v<T>
    ;

template <class T>
inline constexpr bool assert_is_var_expr_v = 
    assert_is_shape_v<T> &&
    assert_var_expr_is_base_of_v<T> &&
    assert_has_type_value_t_v<T> &&
    assert_has_type_index_t_v<T>
    ;

#else

template <class T>
concept var_expr_c = 
    shape_c<T> &&
    var_expr_is_base_of_v<T> &&
    requires () {
        var_expr_traits<T>::has_param;
        var_expr_traits<T>::fixed_size;
        typename var_expr_traits<T>::value_t;
        typename var_expr_traits<T>::index_t;
    } &&
    requires(typename var_expr_traits<T>::index_t offset,
             T& x) {
       { x.set_cache_offset(offset) } -> std::same_as<
               typename var_expr_traits<T>::index_t 
               >;
    } &&
    (
        ( 
            !util::is_mat_v<T> &&
            requires (const MockVector<typename var_expr_traits<T>::value_t>& values,
                      const MockVector< ad::Var<
                        typename var_expr_traits<T>::value_t> >& ad_vars,
                      const T& cx, 
                      size_t i) {
                { cx.value(values, i) } -> std::convertible_to<
                    typename var_expr_traits<T>::value_t>;
                { cx.to_ad(ad_vars, ad_vars, i) } -> ad::is_ad_expr;
            }
        ) ||       
        ( 
            util::is_mat_v<T> &&
            requires (const MockVector<typename var_expr_traits<T>::value_t>& values,
                      const MockVector< ad::Var<
                        typename var_expr_traits<T>::value_t> >& ad_vars,
                      const T& cx, 
                      size_t i) {
                { cx.value(values, i, i) } -> std::convertible_to<
                    typename var_expr_traits<T>::value_t>;
                { cx.to_ad(ad_vars, ad_vars, i, i) } -> ad::is_ad_expr;
            }
        ) 
    )
    ;

template <class T>
concept is_var_expr_v = var_expr_c<T>;

#endif


} // namespace util
} // namespace ppl
