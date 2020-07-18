#pragma once
#if __cplusplus <= 201703L
#include <autoppl/util/traits/concept.hpp>
#else
#include <concepts>
#endif
#include <autoppl/util/traits/type_traits.hpp>
#include <autoppl/util/ad_boost/type_traits.hpp>

namespace ppl {
namespace util {

/**
 * Base class for all model expressions.
 * It is necessary for all model expressions to
 * derive from this class.
 */
template <class T>
struct ModelExprBase : BaseCRTP<T>
{ using BaseCRTP<T>::self; };

/**
 * Checks if ModelExprBase<T> is base of type T 
 */
template <class T>
inline constexpr bool model_expr_is_base_of_v =
    std::is_base_of_v<ModelExprBase<T>, T>;

template <class T>
struct model_expr_traits
{
    using dist_value_t = typename T::dist_value_t;
};

#if __cplusplus <= 201703L

DEFINE_ASSERT_ONE_PARAM(model_expr_is_base_of_v);

template <class T>
inline constexpr bool is_model_expr_v = 
    model_expr_is_base_of_v<T> &&
    has_type_dist_value_t_v<T>
    ;

template <class T>
inline constexpr bool assert_is_model_expr_v = 
    assert_model_expr_is_base_of_v<T> &&
    assert_has_type_dist_value_t_v<T>
    ;

#else

template <class T>
concept model_expr_c =
    model_expr_is_base_of_v<T> &&
    requires (const MockVector<double>& v,
              const MockVector<ad::Var<double>>& ad_vars,
              const T& cx) {
        typename model_expr_traits<T>::dist_value_t;
        { cx.pdf(v) } -> std::same_as<typename model_expr_traits<T>::dist_value_t>;
        { cx.log_pdf(v) } -> std::same_as<typename model_expr_traits<T>::dist_value_t>;
        { cx.ad_log_pdf(ad_vars, ad_vars) } -> ad::is_ad_expr;
    }
    ;

template <class T>
concept is_model_expr_v = model_expr_c<T>;

#endif

} // namespace util
} // namespace ppl
