#pragma once
#include <cassert>
#include <random>
#include <cmath>
#include <numeric>
#include <autoppl/expression/model_expr.hpp>

namespace ppl {

template <typename min_type, typename max_type>
struct Uniform : public ModelExpr<Uniform<min_type, max_type>>
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

    inline value_t min() const { return static_cast<value_t>(min_); }
    inline value_t max() const { return static_cast<value_t>(max_); }

   private:
    min_type min_;
    max_type max_;
};

} // namespace ppl
