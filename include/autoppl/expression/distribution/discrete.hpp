#pragma once
#include <cassert>
#include <random>
#include <cmath>
#include <numeric>

namespace ppl {

// TODO: change name to DiscreteDist and make class template.
// Discrete should be a function that creates this kind of object.

template <typename weight_type>
struct Discrete 
{
    using value_t = int;
    using dist_value_t = double;

    Discrete(std::initializer_list<weight_type> weights)
        : weights_{weights} { assert(weights.size() > 0); }

    template <class GeneratorType>
    value_t sample(GeneratorType& gen) const
    {
        std::discrete_distribution dist(weights_.begin(), weights_.end()); 
        return dist(gen);
    }

    dist_value_t pdf(value_t i) const
    {
        assert( i >= 0 && i < (int) weights_.size() );
        return weights(i) / std::accumulate(weights_.begin(), weights_.end(), 0.0 );

    } 

    dist_value_t log_pdf(value_t i) const
    {
        assert( i >= 0 && i < (int) weights_.size() );
        return log(weights(i) / std::accumulate(weights_.begin(), weights_.end(), 0.0 ));
    }

    inline dist_value_t weights(value_t i) const { return static_cast<dist_value_t>(weights_[i]); }

   private:
    std::vector<weight_type> weights_;
};

} // namespace ppl

