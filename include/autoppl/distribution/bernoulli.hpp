#pragma once
#include <cassert>
#include <random>
#include <cmath>
#include <numeric>
#include <autoppl/distribution/dist_expr.hpp>

namespace ppl {
namespace dist {

template <typename p_type>
struct Bernoulli : public DistExpr<Bernoulli<p_type>>
{
    using value_t = int;
    using dist_value_t = double;

    Bernoulli(p_type p)
        : p_{p} { assert((this -> p() >= 0) && (this -> p() <= 1)); }

    template <class GeneratorType>
    value_t sample(GeneratorType& gen) const
    {
        std::bernoulli_distribution dist(p()); 
        return dist(gen);
    }

    dist_value_t pdf(value_t x) const
    {
        if (x == 1) return p();
        else if (x == 0) return 1. - p();
        else return 0.0;
    }

    dist_value_t log_pdf(value_t x) const
    {
        if (x == 1) return std::log(p());
        else if (x == 0) return std::log(1. - p());
        else return std::numeric_limits<dist_value_t>::lowest();
    }

    dist_value_t p() const { return static_cast<dist_value_t>(p_); }

private:
    p_type p_;
};

} // namespace dist
} // namespace ppl
