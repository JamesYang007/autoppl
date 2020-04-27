#pragma once
#include <cassert>
#include <random>
#include <autoppl/util/var_expr_traits.hpp>
#include <autoppl/util/dist_expr_traits.hpp>

namespace ppl {
namespace expr {

template <typename p_type>
struct Bernoulli : util::DistExpr<Bernoulli<p_type>>
{
    static_assert(util::assert_is_var_expr_v<p_type>);

    using value_t = util::disc_param_t;
    using param_value_t = typename util::var_expr_traits<p_type>::value_t;
    using base_t = util::DistExpr<Bernoulli<p_type>>;
    using dist_value_t = typename base_t::dist_value_t;
    using base_t::pdf;
    using base_t::log_pdf;

    Bernoulli(p_type p)
        : p_{p} {}

    template <class GeneratorType>
    value_t sample(GeneratorType& gen) const
    {
        std::bernoulli_distribution dist(p()); 
        return dist(gen);
    }

    dist_value_t pdf(value_t x, size_t index=0) const
    { 
        if (x == 1) return p(index);
        else if (x == 0) return 1. - p();
        else return 0.0;
    }

    dist_value_t log_pdf(value_t x, size_t index=0) const
    {
        if (x == 1) return std::log(p(index));
        else if (x == 0) return std::log(1. - p(index));
        else return std::numeric_limits<dist_value_t>::lowest();
    }

    param_value_t p(size_t index=0) const { return p_.get_value(index); }
    value_t min() const { return 0; }
    value_t max() const { return 1; }

private:
    p_type p_;
};

} // namespace expr
} // namespace ppl
