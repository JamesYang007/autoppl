#pragma once
#include <cassert>
#include <random>
#include <cmath>
#include <numeric>

namespace ppl {
//namespace expr {
    
// TODO: change name to DiscreteDist and make class template.
// Discrete should be a function that creates this kind of object.

template <typename weight_type>
struct Discrete 
{
    using value_t = uint64_t;
    using dist_value_t = double;

    Discrete() { weights_ = {1}; } 

    Discrete(std::initializer_list<weight_type> weights)
        : weights_{weights} { 
            assert(weights_.size() > 0); 
            assert(all_of(weights_.begin(), weights_.end(), [](weight_type &n){ return n > 0; }));
            double total = std::accumulate(weights_.begin(), weights_.end(), 0.0);
            for_each(weights_.begin(), weights_.end(), [total](weight_type &n){n /= total; });
        }

    template <class Iter>
    Discrete(Iter begin, Iter end)
        :weights_{begin,end} 
    {
        assert(weights_.size() > 0); 
        assert(all_of(weights_.begin(), weights_.end(), [](weight_type &n){ return n > 0; }));
        double total = std::accumulate(weights_.begin(), weights_.end(), 0.0);
        for_each(weights_.begin(), weights_.end(), [total](weight_type &n){n /= total; });
    }

    template <class GeneratorType>
    value_t sample(GeneratorType& gen) const
    {
        std::discrete_distribution dist(weights_.begin(), weights_.end()); 
        return dist(gen); 
    }

    dist_value_t pdf(value_t i) const
    {
        assert( i >= 0 && i <  weights_.size() );
        return weights(i) ;

    } 

    dist_value_t log_pdf(value_t i) const
    {
        assert( i >= 0 && i < weights_.size() );
        return std::log(weights(i));
    }

    inline dist_value_t weights(value_t i) const { return static_cast<dist_value_t>(weights_[i]); }

   private:
    std::vector<weight_type> weights_;
};

template<typename Iter> Discrete(Iter,Iter) -> Discrete<typename Iter::value_type>;

//} // namespace expr
} // namespace ppl

