#pragma once
#include <type_traits>

namespace ppl {

template <class Derived>
struct VarExpr
{
    Derived& self() 
    { return static_cast<Derived&>(*this); }

    const Derived& self() const
    { return static_cast<const Derived&>(*this); }
};

template <class T>
inline constexpr bool is_var_expr_v = 
    std::is_convertible_v<T, VarExpr<T>>;

#ifdef AUTOPPL_USE_CONCEPTS
// TODO: definition should be extended with a stronger
// restriction on T with interface checking.
template <class T>
concept var_expressable = is_var_expr_v<T>; 
#endif

} // namespace ppl
