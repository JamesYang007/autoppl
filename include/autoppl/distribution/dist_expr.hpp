#pragma once
#include <type_traits>

namespace ppl {
namespace dist {

template <class Derived>
struct DistExpr
{
    Derived& self()
    { return static_cast<Derived&>(*this); }

    const Derived& self() const
    { return static_cast<const Derived&>(*this); }
};

template <class T>
inline constexpr bool is_dist_expr_v = 
    std::is_convertible_v<T, dist::DistExpr<T>>;

#ifdef AUTOPPL_USE_CONCEPTS
// TODO: definition should be extended with a stronger
// restriction on T with interface checking.
template <class T>
concept dist_expressable = is_dist_expr_v<T>; 
#endif

} // namespace dist
} // namespace ppl
