#pragma once
#include <cmath>
#include <numeric>

namespace ppl {
namespace expr {

/*
 * The Base objects contain static member functions 
 * that compute pdf and log_pdf.
 * These are useful stand-alone functions and the distribution objects
 * such as Uniform and Normal simply wrap these functions.
 */

/*
 * Continuous Distributions
 */

struct UniformBase
{
    using dist_value_t = double;

    template <class ValueType, class ParamValueType>
    static dist_value_t pdf(ValueType x,
                            ParamValueType min,
                            ParamValueType max)
    {
        return (min < x && x < max) ? 1. / (max - min) : 0;
    }

    template <class ValueType, class ParamValueType>
    static dist_value_t log_pdf(ValueType x,
                                ParamValueType min,
                                ParamValueType max)
    {
        return (min < x && x < max) ? 
            -std::log(max - min) : 
            std::numeric_limits<dist_value_t>::lowest();
    }
};

struct NormalBase
{
    using dist_value_t = double;

    template <class ValueType, class ParamValueType>
    static dist_value_t pdf(ValueType x,
                            ParamValueType mean,
                            ParamValueType stddev)
    {
        dist_value_t z_score = (x - mean) / stddev;
        return std::exp(- 0.5 * z_score * z_score) / (stddev * std::sqrt(2 * M_PI));
    }

    template <class ValueType, class ParamValueType>
    static dist_value_t log_pdf(ValueType x,
                                ParamValueType mean,
                                ParamValueType stddev)
    {
        dist_value_t z_score = (x - mean) / stddev;
        return -0.5 * ((z_score * z_score) + std::log(stddev * stddev * 2 * M_PI));
    }
};

/*
 * Discrete Distributions
 */

struct BernoulliBase
{
    using dist_value_t = double;

    template <class ValueType, class ParamValueType>
    static dist_value_t pdf(ValueType x, ParamValueType p)
    { 
        if (x == 1) return p;
        else if (x == 0) return 1. - p;
        else return 0.0;
    }

    template <class ValueType, class ParamValueType>
    static dist_value_t log_pdf(ValueType x, ParamValueType p)
    {
        if (x == 1) return std::log(p);
        else if (x == 0) return std::log(1. - p);
        else return std::numeric_limits<dist_value_t>::lowest();
    }
};

} // namespace expr
} // namespace ppl
