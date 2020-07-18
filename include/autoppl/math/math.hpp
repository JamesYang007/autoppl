#pragma once
#include <cmath>
#include <algorithm>
#include <iterator>
#include <autoppl/util/functional.hpp>

namespace ppl {
namespace math {

template <class T>
inline constexpr T inf = 
    std::numeric_limits<T>::is_iec559 ? 
    std::numeric_limits<T>::infinity() :
    std::numeric_limits<T>::max();

template <class T>
inline constexpr T neg_inf = 
    std::numeric_limits<T>::is_iec559 ? 
    -std::numeric_limits<T>::infinity() :
    std::numeric_limits<T>::lowest();

template <class Iter, class F = util::identity>
inline constexpr auto min(Iter begin, Iter end, F f = F())
{
    using value_t = typename std::iterator_traits<Iter>::value_type;
    static_assert(std::is_invocable_v<F, value_t>);
    using ret_value_t = std::decay_t<
        decltype(f(std::declval<value_t>())) >;

    if (std::distance(begin, end) <= 0) {
        return inf<ret_value_t>;
    } 

    ret_value_t res = inf<ret_value_t>;
    std::for_each(begin, end, 
                  [&](value_t x) 
                  { res = std::min(res, f(x)); });
    return res;
}

template <class Iter, class F = util::identity>
inline constexpr auto max(Iter begin, Iter end, F f = F())
{
    using value_t = typename std::iterator_traits<Iter>::value_type;
    static_assert(std::is_invocable_v<F, value_t>);
    using ret_value_t = std::decay_t<
        decltype(f(std::declval<value_t>())) >;

    if (std::distance(begin, end) <= 0) {
        return neg_inf<ret_value_t>;
    } 

    ret_value_t res = neg_inf<ret_value_t>;
    std::for_each(begin, end, 
                  [&](value_t x) 
                  { res = std::max(res, f(x)); });
    return res;
}

/**
 * LogSumExp taken from wikipedia: log(e^x + e^y)
 */
template <class T>
inline T lse(T x, T y)
{
    if (x >= y) return x + std::log(1. + std::exp(y-x));
    else return lse(y, x);
}

} // namespace math
} // namespace ppl
