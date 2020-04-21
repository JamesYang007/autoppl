#pragma once
#include <cstdint>
#include <autoppl/util/concept.hpp>

namespace ppl {
namespace util {

/* 
 * TODO: Samplable distribution expression concept?
 */

/* 
 * TODO: continuous/discrete distribution expression concept?
 */

/*
 * Continuous distribution expressions can be constructed with this type.
 */
using cont_param_t = double;

/*
 * Discrete distribution expressions can be constructed with this type.
 */
using disc_param_t = int64_t;

/*
 * Traits for Distribution Expression classes.
 * value_t      type of value Variable represents during computation
 * dist_value_t type of pdf/log_pdf value
 */
template <class DistExprType>
struct dist_expr_traits
{
    using value_t = typename DistExprType::value_t;
    using dist_value_t = typename DistExprType::dist_value_t;
};

/*
 * A distribution expression is any class that satisfies the following concept:
 */
template <class T>
inline constexpr bool is_dist_expr_v = 
    has_type_value_t_v<T> &&
    has_type_dist_value_t_v<T> &&
    has_func_pdf_v<const T> &&
    has_func_log_pdf_v<const T>
    ;

template <class T>
inline constexpr bool assert_is_dist_expr_v = 
    assert_has_type_value_t_v<T> &&
    assert_has_type_dist_value_t_v<T> &&
    assert_has_func_pdf_v<const T> &&
    assert_has_func_log_pdf_v<const T>
    ;

} // namespace util
} // namespace ppl
