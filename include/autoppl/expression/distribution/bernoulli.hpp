#pragma once
#include <cassert>
#include <random>
#include <autoppl/util/var_expr_traits.hpp>
#include <autoppl/util/dist_expr_traits.hpp>
#include <autoppl/expression/distribution/base.hpp>

namespace ppl {
namespace expr {

template <typename p_type>
struct Bernoulli
{
    static_assert(util::is_var_expr_v<p_type>);

    using value_t = util::disc_param_t;
    using param_value_t = typename util::var_expr_traits<p_type>::value_t;
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

} // namespace expr
} // namespace ppl
