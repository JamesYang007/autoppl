#pragma once
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
    
namespace ppl {

/*
 * Checks if type From can be explicitly converted to type To.
 */
template <class From, class To>
inline constexpr bool is_explicitly_convertible_v =
    std::is_constructible_v<To, From> &&
    !std::is_convertible_v<From, To>
    ;
DEFINE_ASSERT_TWO_PARAM(is_explicitly_convertible_v);

} // namespace ppl
