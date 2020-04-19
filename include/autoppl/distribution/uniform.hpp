#pragma once
#include <cassert>
#include <random>
#include <cmath>
#include <numeric>
#include <autoppl/distribution/dist_expr.hpp>

namespace ppl {
namespace dist {

template <typename min_type, typename max_type>
struct Uniform : public DistExpr<Uniform<min_type, max_type>>
{
    using value_t = double;
    using dist_value_t = double;

    Uniform(min_type min, max_type max)
        : min_{min}, max_{max} { assert(this -> min() < this -> max()); }

    // TODO: tag this class as "TriviallySamplable"?
    template <class GeneratorType>
    value_t sample(GeneratorType& gen) const
    {
        std::uniform_real_distribution<value_t> dist(min(), max()); 
        return dist(gen);
    }

    dist_value_t pdf(value_t x) const
    {
        return (min() < x && x < max()) ? 1. / (max() - min()) : 0;
    }

    dist_value_t log_pdf(value_t x) const
    {
        return (min() < x && x < max()) ? 
            -std::log(max() - min()) : 
            std::numeric_limits<dist_value_t>::lowest();
    }

    value_t min() const { return static_cast<value_t>(min_); }
    value_t max() const { return static_cast<value_t>(max_); }

private:
    min_type min_;
    max_type max_;
};

} // namespace dist
} // namespace ppl
