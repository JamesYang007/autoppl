#pragma once
#if __cplusplus <= 201703L
#include <autoppl/util/traits/concept.hpp>
#else
#include <concepts>
#endif
#include <autoppl/util/traits/type_traits.hpp>

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

template <class T>
inline constexpr bool is_model_expr_v = 
    model_expr_is_base_of_v<T> &&
    has_type_dist_value_t_v<T>
    ;

#else

template <class T>
concept model_expr_c =
    model_expr_is_base_of_v<T> &&
    requires (const T& cx) {
        typename model_expr_traits<T>::dist_value_t;
        { cx.pdf() } -> std::same_as<typename model_expr_traits<T>::dist_value_t>;
        { cx.log_pdf() } -> std::same_as<typename model_expr_traits<T>::dist_value_t>;
    }
    ;

template <class T>
concept is_model_expr_v = model_expr_c<T>;

#endif

} // namespace util
} // namespace ppl
