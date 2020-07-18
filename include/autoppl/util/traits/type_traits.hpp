#pragma once
#include <cstddef>
#include <type_traits>

#define DEFINE_ASSERT_ONE_PARAM(name) \
    namespace details {                         \
        template <bool b>                       \
        struct assert_##name                    \
        {                                       \
            static_assert(b,                    \
                    "Assert "#name" failed"  \
                    );                          \
        };                                      \
                                                \
        template<>                              \
        struct assert_##name<true> : std::true_type \
        {};                                     \
    }                                           \
    template <class T>                 \
    inline constexpr bool assert_##name =       \
        details::assert_##name<name<T>>::value; \

#define DEFINE_ASSERT_TWO_PARAM(name) \
    namespace details {                         \
        template <bool b>                       \
        struct assert_##name                    \
        {                                       \
            static_assert(b,                    \
                    "Assert "#name" failed"  \
                    );                          \
        };                                      \
                                                \
        template<>                              \
        struct assert_##name<true> : std::true_type \
        {};                                     \
    }                                           \
    template <class T, class U>                 \
    inline constexpr bool assert_##name =       \
        details::assert_##name<name<T, U>>::value; \
    
// Important type checking error messages
#define PPL_CONT_XOR_DISC \
    "Expression must be either continuous or discrete. " \
    "It cannot be both continuous and discrete. "

namespace ppl {
namespace util {

/**
 * Checks if type From can be explicitly converted to type To.
 */
template <class From, class To>
inline constexpr bool is_explicitly_convertible_v =
    std::is_constructible_v<To, From> &&
    !std::is_convertible_v<From, To>
    ;
DEFINE_ASSERT_TWO_PARAM(is_explicitly_convertible_v);

/**
 * Used for CRTP to unify certain expression types under one name.
 * CRTP types should simply derive from this base class.
 */
template <class T>
struct BaseCRTP 
{
    T& self() { return static_cast<T&>(*this); }
    const T& self() const { return static_cast<const T&>(*this); }
};

template <class T>
inline constexpr bool is_cont_v = std::is_floating_point_v<T>;

template <class T>
inline constexpr bool is_disc_v = std::is_integral_v<T>;

/**
 * Mock types used to check concepts
 */

/**
 * MockVector satisfies the following properties:
 * - operator[](size_t) defined (return value does not matter)
 */
template <class T>
struct MockVector
{
    T& operator[](size_t);
    const T& operator[](size_t) const;
};


} // namespace util
} // namespace ppl
