#pragma once
#if __cplusplus <= 201703L
#include <autoppl/util/concept.hpp>
#endif
#include <autoppl/util/type_traits.hpp>

namespace ppl {
namespace util {

/**
 * Base class for all model expressions.
 * It is necessary for all model expressions to
 * derive from this class.
 */
template <class T>
struct ModelExpr : BaseCRTP<T>
{ using BaseCRTP<T>::self; };

/**
 * Checks if DistExpr<T> is base of type T 
 */
template <class T>
inline constexpr bool model_expr_is_base_of_v =
    std::is_base_of_v<ModelExpr<T>, T>;

#if __cplusplus <= 201703L
DEFINE_ASSERT_ONE_PARAM(model_expr_is_base_of_v);
#endif

/**
 * Traits for Model Expression classes.
 * dist_value_t      type of value Variable represents during computation
 */
template <class NodeType>
struct model_expr_traits
{
    using dist_value_t = typename NodeType::dist_value_t;
};

#if __cplusplus <= 201703L

// TODO: 
// - pdf and log_pdf remove from interface?
// - how to check if template member function exists (for traverse)?
template <class T>
inline constexpr bool is_model_expr_v = 
    model_expr_is_base_of_v<T> &&
    has_type_dist_value_t_v<T> &&
    has_func_pdf_v<const T> &&
    has_func_log_pdf_v<const T>
    ;

template <class T>
inline constexpr bool assert_is_model_expr_v = 
    assert_model_expr_is_base_of_v<T> &&
    assert_has_type_dist_value_t_v<T> &&
    assert_has_func_pdf_v<const T> &&
    assert_has_func_log_pdf_v<const T>
    ;

#else

template <class T>
concept model_expr =
    model_expr_is_base_of_v<T> &&
    requires (const T cx) {
        typename model_expr_traits<T>::dist_value_t;
        {cx.pdf()} -> std::same_as<typename model_expr_traits<T>::dist_value_t>;
        {cx.log_pdf()} -> std::same_as<typename model_expr_traits<T>::dist_value_t>;
    }
    ;

#endif

} // namespace util
} // namespace ppl
