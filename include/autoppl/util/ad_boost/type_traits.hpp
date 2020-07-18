#pragma once
#include <type_traits>
#include <fastad>

namespace ad {

/**
 * Checks if a given type is an AD expression type.
 */
namespace details {

template <class T>
struct is_ad_expr 
{
    static constexpr bool value =
        std::is_base_of_v<ad::core::ADNodeExpr<T>, T>;
};

} // namespace details

template <class T>
inline constexpr bool is_ad_expr_v =
    details::is_ad_expr<T>::value;

#if __cplusplus > 201703L

template <class T>
concept is_ad_expr =
    details::is_ad_expr<T>::value;

#endif

} // namespace ad

