#pragma once
#include <type_traits>

/**
 * Metaprogramming tools to check if name is a (public) 
 * member alias of a given type T.
 * All instances of this macro must be placed in this file for ease of maintenance.
 * Macro definition is undefined at the end of the file.
 *
 * has_type_name_v simply checks if type T has the member alias "name".
 * It always returns either true or false with no compiler-error,
 * so long as T is resolved to a valid type.
 *
 * get_type_name_t returns T::name if "name" is a member alias of T,
 * and otherwise returns type invalid_tag.
 *
 * assert_has_type_name_v asserts that type T has the member alias "name".
 * If "name" is not a member alias, it is a compiler error 
 * and an appropriate message is printed.
 *
 * Ex. with "name" as "value_t"
 *
    namespace details {                         \
        template<class T>                       \
        struct has_type_value_t                  \
        {                                       \
        private:                                \
            template<typename V> static void impl(typename V::value_t*); \
            template<typename V> static bool impl(...);                                \
        public:                                                                         \
            static constexpr bool value = std::is_same<void, decltype(impl<T>(0))>::value;  \
        };                                      \
                                                \
        template <class T, bool = false>        \
        struct get_type_value_t                  \
        {                                       \
            using type = invalid_tag;           \
        };                                      \
        template <class T>                      \
        struct get_type_value_t<T, true>         \
        {                                       \
            using type = typename T::value_t;      \
        };                                      \
                                                \
        template <bool b>                       \
        struct assert_has_type_value_t           \
        {                                       \
            static_assert(b, "Type does not have member type ""value_t"); \
        };                                      \
                                                \
        template<>                              \
        struct assert_has_type_value_t<true> : std::true_type \
        {};                                     \
    }                                           \
    template <class T>                          \
    inline constexpr bool has_type_value_t_v = \
        details::has_type_value_t<T>::value;     \
    template <class T>                          \
    inline constexpr bool assert_has_type_value_t_v = \
        details::assert_has_type_value_t<has_type_value_t_v<T>>::value; \
    template <class T>                          \
    using get_type_value_t_t =                 \
        typename details::get_type_value_t<T, has_type_value_t_v<T>>::type;
 */

#define DEFINE_HAS_TYPE(name)                   \
    namespace details {                         \
        template<class T>                       \
        struct has_type_##name                  \
        {                                       \
        private:                                \
            template<typename V> static void impl(typename V::name*); \
            template<typename V> static bool impl(...);                                \
        public:                                                                         \
            static constexpr bool value = std::is_same<void, decltype(impl<T>(0))>::value;  \
        };                                      \
                                                \
        template <class T, bool = false>        \
        struct get_type_##name                  \
        {                                       \
            using type = invalid_tag;           \
        };                                      \
        template <class T>                      \
        struct get_type_##name<T, true>         \
        {                                       \
            using type = typename T::name;      \
        };                                      \
                                                \
        template <bool b>                       \
        struct assert_has_type_##name           \
        {                                       \
            static_assert(b, "Type does not have member type "#name); \
        };                                      \
                                                \
        template<>                              \
        struct assert_has_type_##name<true> : std::true_type \
        {};                                     \
    }                                           \
    template <class T>                          \
    inline constexpr bool has_type_##name##_v = \
        details::has_type_##name<T>::value;     \
    template <class T>                          \
    inline constexpr bool assert_has_type_##name##_v = \
        details::assert_has_type_##name<has_type_##name##_v<T>>::value; \
    template <class T>                          \
    using get_type_##name##_t =                 \
        typename details::get_type_##name<T, has_type_##name##_v<T>>::type;


/**
 * Metaprogramming tool to check if name is a (public) member function of a given type T.
 * All instances of this macro must be placed in this file for ease of maintenance.
 * Macro definition is undefined at the end of the file.
 *
 * has_func_name_v simply checks if type T has the public, non-overloaded
 * member function "name".
 * It always returns either true or false with no compiler-error,
 * so long as T is resolved to a valid type.
 *
 * assert_has_func_name_v asserts that type T has the public, non-overloaded
 * member function "name".
 * If "name" is not such function, it is a compiler error 
 * and an appropriate message is printed.
 *
 * Ex. with "name" as "pdf"
 *
    namespace details {                         \
        template<class T>                       \
        struct has_func_pdf                  \
        {                                       \
        private:                                \
            template<typename V> static void impl(decltype(&V::pdf)); \
            template<typename V> static bool impl(...);                                \
        public:                                                                         \
            static constexpr bool value = std::is_same<void, decltype(impl<T>(0))>::value;  \
        };                                      \
                                                \
        template <bool b>                       \
        struct assert_has_func_pdf           \
        {                                       \
            static_assert(b,                    \
                    "Type does not have public, non-overloaded " \
                    "member function ""pdf"     \
                    );                          \
        };                                      \
                                                \
        template<>                              \
        struct assert_has_func_pdf<true> : std::true_type \
        {};                                     \
    }                                           \
    template <class T>                          \
    inline constexpr bool has_func_pdf_v =   \
        details::has_func_pdf<T>::value;     \
    template <class T>                          \
    inline constexpr bool assert_has_func_pdf_v = \
        details::assert_has_func_pdf<has_func_pdf_v<T>>::value; \
 */

#define DEFINE_HAS_FUNC(name)                   \
    namespace details {                         \
        template<class T>                       \
        struct has_func_##name                  \
        {                                       \
        private:                                \
            template<typename V> static void impl(decltype(&V::name)); \
            template<typename V> static bool impl(...);                                \
        public:                                                                         \
            static constexpr bool value = std::is_same<void, decltype(impl<T>(0))>::value;  \
        };                                      \
                                                \
        template <bool b>                       \
        struct assert_has_func_##name           \
        {                                       \
            static_assert(b,                    \
                    "Type does not have public, non-overloaded " \
                    "member function "#name     \
                    );                          \
        };                                      \
                                                \
        template<>                              \
        struct assert_has_func_##name<true> : std::true_type \
        {};                                     \
    }                                           \
    template <class T>                          \
    inline constexpr bool has_func_##name##_v =   \
        details::has_func_##name<T>::value;     \
    template <class T>                          \
    inline constexpr bool assert_has_func_##name##_v = \
        details::assert_has_func_##name<has_func_##name##_v<T>>::value; \

namespace ppl {
namespace util {

/**
 * The type invalid_tag is used as a "black hole" 
 * for when a condition is not met, but cannot set compiler error.
 */
struct invalid_tag
{
    invalid_tag() =delete;
    ~invalid_tag() =delete;
    invalid_tag(const invalid_tag&) =delete;
    invalid_tag& operator=(const invalid_tag&) =delete;
    invalid_tag(invalid_tag&&) =delete;
    invalid_tag& operator=(invalid_tag&&) =delete;
};

DEFINE_HAS_TYPE(value_t);
DEFINE_HAS_TYPE(pointer_t);
DEFINE_HAS_TYPE(const_pointer_t);

DEFINE_HAS_TYPE(dist_value_t);

DEFINE_HAS_FUNC(set_value);
DEFINE_HAS_FUNC(get_value);
DEFINE_HAS_FUNC(set_storage);
DEFINE_HAS_FUNC(get_storage);

DEFINE_HAS_FUNC(pdf);
DEFINE_HAS_FUNC(log_pdf);
DEFINE_HAS_FUNC(min);
DEFINE_HAS_FUNC(max);

DEFINE_HAS_FUNC(get_variable);
DEFINE_HAS_FUNC(get_distribution);

} // namespace util
} // namespace ppl

#undef DEFINE_HAS_FUNC
#undef DEFINE_HAS_TYPE
