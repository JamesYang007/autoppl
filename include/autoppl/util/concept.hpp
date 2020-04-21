#pragma once
#include <type_traits>

/*
 * Metaprogramming tool to check if name is a (public) 
 * member alias of a given type T.
 * All versions must be placed in this file for ease of maintenance.
 * Macro definition is undefined at the end of the file.
 *
 * Ex. with "name" as "value_t"
 *
   namespace details {                         
       template<class T>                       
       struct has_type_value_t
       {                                       
       private:                                
           template<typename V> static void impl(decltype(typename V::value_t(), int())); 
           template<typename V> static bool impl(char);                                
       public:                                                                         
           static constexpr bool value = std::is_same<void, decltype(impl<T>(0))>::value;  
       };                                      
   }                                           
   template <class T>                          
   inline constexpr bool has_type_value_t_v = 
       details::has_type_value_t<T>::value;
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
    }                                           \
    template <class T>                          \
    inline constexpr bool has_type_##name##_v = \
        details::has_type_##name<T>::value;     \
    template <class T>                          \
    using get_type_##name##_t =                 \
        typename details::get_type_##name<T, has_type_##name##_v<T>>::type;

/*
 * Metaprogramming tool to check if name is a (public) 
 * member function of a given type T.
 * All versions must be placed in this file for ease of maintenance.
 * Macro definition is undefined at the end of the file.
 *
 * Ex. with "name" as "pdf"
 *
   namespace details {                         
       template<class T>                       
       struct has_func_pdf
       {                                       
       private:                                
           template<typename V> static void impl(decltype(&V::pdf)); 
           template<typename V> static bool impl(...);                       
       public:                                                                         
           static constexpr bool value = std::is_same<void, decltype(impl<T>(0))>::value;  
       };                                      
   }                                           
   template <class T>                          
   inline constexpr bool has_func_pdf_v = 
       details::has_func_pdf<T>::value;
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
    }                                           \
    template <class T>                          \
    inline constexpr bool has_func_##name##_v =   \
        details::has_func_##name<T>::value;

namespace ppl {
namespace util {

struct invalid_tag {};

DEFINE_HAS_TYPE(value_t);
DEFINE_HAS_TYPE(pointer_t);
DEFINE_HAS_TYPE(const_pointer_t);
DEFINE_HAS_TYPE(state_t);

DEFINE_HAS_TYPE(dist_value_t);

DEFINE_HAS_FUNC(set_value);
DEFINE_HAS_FUNC(get_value);
DEFINE_HAS_FUNC(set_storage);
DEFINE_HAS_FUNC(get_storage);
DEFINE_HAS_FUNC(set_state);
DEFINE_HAS_FUNC(get_state);

DEFINE_HAS_FUNC(pdf);
DEFINE_HAS_FUNC(log_pdf);

DEFINE_HAS_FUNC(get_variable);
DEFINE_HAS_FUNC(get_distribution);

} // namespace util
} // namespace ppl

#undef DEFINE_HAS_FUNC
#undef DEFINE_HAS_TYPE
