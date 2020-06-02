#pragma once
#include <cmath>

namespace ppl {
namespace math {

/**
 * LogSumExp taken from wikipedia:
 * log(e^x + e^y)
 */
template <class T>
inline T lse(T x, T y)
{
    return std::log(std::exp(x) + std::exp(y));
}

} // namespace math
} // namespace ppl
