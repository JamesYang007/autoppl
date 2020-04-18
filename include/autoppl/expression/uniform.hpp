#pragma once
#include <cassert>
#include <random>
#include <cmath>
#include <numeric>

namespace ppl {

// TODO: change name to UniformDist and make class template.
// uniform should be a function that creates this kind of object.

template <typename min_type, typename max_type>
struct Uniform 
{
    using value_t = double;
    using dist_value_t = double;

    Uniform(min_type min, max_type max)
        : min_{min}, max_{max} { assert(static_cast<value_t>(min_) < static_cast<value_t>(max_)); }

    // TODO: tag this class as "TriviallySamplable"?
    template <class GeneratorType>
    value_t sample(GeneratorType& gen) const
    {
        value_t min, max;
        min = static_cast<value_t>(min_);
        max = static_cast<value_t>(max_);

        std::uniform_real_distribution<value_t> dist(min, max); 
        return dist(gen);
    }

    dist_value_t pdf(value_t x) const
    {
        value_t min, max;
        min = static_cast<value_t>(min_);
        max = static_cast<value_t>(max_); 
        
        return (min < x && x < max) ? 1. / (max - min) : 0;
    }

    dist_value_t log_pdf(value_t x) const
    {
        value_t min, max;
        min = static_cast<value_t>(min_);
        max = static_cast<value_t>(max_);

        return (min < x && x < max) ? 
            -std::log(max - min) : 
            std::numeric_limits<dist_value_t>::lowest();
    }

private:
    min_type min_;
    max_type max_;
};

} // namespace ppl
