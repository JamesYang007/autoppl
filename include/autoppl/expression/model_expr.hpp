#pragma once
#include <type_traits>

namespace ppl {
namespace expr {

template <class Derived>
struct ModelExpr
{
    Derived& self() 
    { return static_cast<Derived&>(*this); }

    const Derived& self() const
    { return static_cast<const Derived&>(*this); }
};

template <class T>
inline constexpr bool is_model_expr_v = 
    std::is_convertible_v<T, ModelExpr<T>>;

#ifdef AUTOPPL_USE_CONCEPTS
// TODO: definition should be extended with a stronger
// restriction on T with interface checking.
template <class T>
concept model_expressable = is_model_expr_v<T>; 
#endif

} // namespace expr
} // namespace ppl
