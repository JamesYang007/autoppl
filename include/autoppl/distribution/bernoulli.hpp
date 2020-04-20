#pragma once
#include <cassert>
#include <random>
#include <cmath>
#include <numeric>
#include <autoppl/util/traits.hpp>
#include <autoppl/distribution/dist_expr.hpp>
#include <autoppl/distribution/density.hpp>

namespace ppl {
namespace dist {

template <typename p_type>
struct Bernoulli : public DistExpr<Bernoulli<p_type>>
{
    using value_t = uint8_t;
    using param_value_t = typename var_traits<p_type>::value_t;
    using dist_value_t = typename BernoulliBase::dist_value_t;

    Bernoulli(p_type p)
        : p_{p} { assert((this -> p() >= 0) && (this -> p() <= 1)); }

    template <class GeneratorType>
    value_t sample(GeneratorType& gen) const
    {
        std::bernoulli_distribution dist(p()); 
        return dist(gen);
    }

    dist_value_t pdf(value_t x) const
    { return BernoulliBase::pdf(x, p()); }

    dist_value_t log_pdf(value_t x) const
    { return BernoulliBase::log_pdf(x, p()); }

    param_value_t p() const { return static_cast<param_value_t>(p_); }

private:
    p_type p_;
};

} // namespace dist
} // namespace ppl
