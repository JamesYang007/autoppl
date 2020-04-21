#pragma once
#include <type_traits>

namespace ppl {

/*
 * Checks if type From can be explicitly converted to type To.
 */
template <class From, class To>
inline constexpr bool is_explicitly_convertible_v =
    std::is_constructible_v<To, From> &&
    !std::is_convertible_v<From, To>
    ;

} // namespace ppl
