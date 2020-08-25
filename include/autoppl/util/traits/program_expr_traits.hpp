#pragma once
#include <autoppl/util/traits/type_traits.hpp>

namespace ppl {
namespace util {

template <class Derived>
struct ProgramExprBase: BaseCRTP<Derived>
{};

template <class T>
inline constexpr bool program_expr_is_base_of_v =
    std::is_base_of_v<ProgramExprBase<T>, T>;

template <class T>
struct program_expr_traits
{};

#if __cplusplus <= 201703L

template <class T>
inline constexpr bool is_program_expr_v = 
    program_expr_is_base_of_v<T>
    ;

#else

template <class T>
concept program_expr_c =
    program_expr_is_base_of_v<T>
    ;

template <class T>
concept is_program_expr_v = program_expr_c<T>;

#endif

} // namespace util
} // namespace ppl
