#pragma once
#include <cassert>
#include <random>
#include <cmath>
#include <numeric>

namespace ppl {

struct uniform 
{
    using value_t = double;
    using dist_value_t = double;

    uniform(value_t min, value_t max)
        : min_{min}, max_{max}
    {
        assert(min_ < max_);
    }

    // TODO: tag this class as "TriviallySamplable"
    template <class GeneratorType>
    value_t sample(GeneratorType& gen) const
    {
        std::uniform_real_distribution<value_t> dist(min_, max_); 
        return dist(gen);
    }

    dist_value_t pdf(value_t x) const
    {
        return (min_ < x && x < max_) ? 1./(max_ - min_) : 0;
    }

    dist_value_t log_pdf(value_t x) const
    {
        return (min_ < x && x < max_) ? 
            -std::log(max_ - min_) : 
            std::numeric_limits<dist_value_t>::lowest();
    }

private:
    value_t min_, max_;
};

} // namespace ppl
