#pragma once
#include <cmath>
#include <autoppl/math/math.hpp>

// MSVC does not seem to support M_PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ppl {
namespace math {

/////////////////////////////////
// Compile-time Constants
/////////////////////////////////

inline constexpr double SQRT_TWO_PI = 
    2.506628274631000502415765284811045;
inline constexpr double LOG_SQRT_TWO_PI =
    0.918938533204672741780329736405617;

/////////////////////////////////
// Univariate densities
/////////////////////////////////

template <class T>
inline constexpr T normal_pdf(T x, T mean, T sd)
{
    T z_score = (x - mean) / sd;
    return std::exp(-0.5 * z_score * z_score) / 
        (sd * SQRT_TWO_PI);
}

template <class T>
inline constexpr T normal_log_pdf(T x, T mean, T sd)
{
    T z_score = (x - mean) / sd;
    return (-0.5 * z_score * z_score) - std::log(sd) - LOG_SQRT_TWO_PI;
}

template <class T>
inline constexpr T uniform_pdf(T x, T min, T max)
{
    return (min < x && x < max) ? 1. / (max - min) : 0;
}

template <class T>
inline constexpr T uniform_log_pdf(T x, T min, T max)
{
    return (min < x && x < max) ? 
        -std::log(max - min) : 
        neg_inf<T>;
}

/**
 * Bernoulli pdf and log pdf (pmf actually).
 * It is defined to clip when p is out of the range [0,1],
 * i.e. if p < 0, then we take p = 0 and
 * if p > 1, then we take p = 1.
 */
template <class IntType, class T>
inline constexpr T bernoulli_pdf(IntType x, T p)
{
    if (p <= 0) return x == 0;
    else if (p >= 1) return x == 1;

    if (x == 1) return p;
    else if (x == 0) return 1. - p;
    else return 0.0;
}

template <class IntType, class T>
inline constexpr T bernoulli_log_pdf(IntType x, T p)
{
    if (p <= 0) {
        if (x == 0) return 0;
        else return neg_inf<T>;
    }
    else if (p >= 1) {
        if (x == 1) return 0;
        else return neg_inf<T>;
    }

    if (x == 1) return std::log(p);
    else if (x == 0) return std::log(1. - p);
    else return neg_inf<T>;
}

} // namespace math
} // namespace ppl
