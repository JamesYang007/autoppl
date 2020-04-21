#pragma once
#include <cassert>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace ppl {
namespace expr {
    
// TODO: change name to DiscreteDist and make class template.
// Discrete should be a function that creates this kind of object.

template <typename weight_type>
struct Discrete 
{
    using value_t = uint64_t;
    using dist_value_t = double;

    Discrete() { weights_ = {1}; } 

    Discrete(std::initializer_list<weight_type> weights)
        : weights_{ normalize_weights(weights) } {  }

    template <class Iter>
    Discrete(Iter begin, Iter end)
        :weights_{ normalize_weights(std::vector<weight_type> (begin,end)) } { }

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
    std::vector<weight_type> normalize_weights(std::vector<weight_type> w){
        // check that weights are positive, not empty, and normalize the weights
        assert(w.size() > 0); 
        assert(std::all_of(w.begin(), w.end(), [](weight_type &n){ return n > 0; }));
        double total = std::accumulate(w.begin(), w.end(), 0.0);
        std::for_each(w.begin(), w.end(), [total](weight_type &n){n /= total; }); 
        return w;
    }
};

template<typename Iter> Discrete(Iter,Iter) -> Discrete<typename Iter::value_type>;

} // namespace expr
} // namespace ppl

