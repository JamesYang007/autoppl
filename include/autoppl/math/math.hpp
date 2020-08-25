#pragma once
#include <cmath>
#include <algorithm>
#include <iterator>

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

/**
 * LogSumExp taken from wikipedia: log(e^x + e^y)
 */
template <class T>
inline T lse(T x, T y)
{
    if (x == neg_inf<T>) return y;
    if (x == inf<T> && y == inf<T>) return inf<T>;
    if (x >= y) return x + std::log(1. + std::exp(y-x));
    else return lse(y, x);
}

} // namespace math
} // namespace ppl
