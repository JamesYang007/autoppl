#pragma once
#include <autoppl/util/concept.hpp>

namespace ppl {
namespace util {

/*
 * Traits for Model Expression classes.
 * dist_value_t      type of value Variable represents during computation
 */
template <class NodeType>
struct model_expr_traits
{
    using dist_value_t = typename NodeType::dist_value_t;
};

// TODO: 
// - pdf and log_pdf remove from interface?
// - how to check if template member function exists (for traverse)?
template <class T>
inline constexpr bool is_model_expr_v = 
    has_type_dist_value_t_v<T> &&
    has_func_pdf_v<const T> &&
    has_func_log_pdf_v<const T>
    ;

// TODO: not used currently
template <class T>
inline constexpr bool is_eq_node_expr_v =
    is_model_expr_v<T> && 
    has_func_get_variable_v<T> &&
    has_func_get_distribution_v<T>
    ;   

#ifdef AUTOPPL_USE_CONCEPTS
// TODO: definition should be extended with a stronger
// restriction on T with interface checking.
template <class T>
concept model_expressable = is_model_expr_v<T>; 
#endif

} // namespace util
} // namespace ppl
