#pragma once
#include <cassert>
#include <random>
#include <cmath>
#include <numeric>
#include <autoppl/distribution/dist_expr.hpp>

namespace ppl {
namespace dist {

template <typename mean_type, typename var_type>
struct Normal : public DistExpr<Normal<mean_type, var_type>>
{
    using value_t = double;
    using dist_value_t = double;

    static_assert(std::is_convertible_v<mean_type, value_t>);
    static_assert(std::is_convertible_v<var_type, value_t>);

    Normal(mean_type mean, var_type var)
        : mean_{mean}, var_{var} {
            assert(this -> var() > 0);
        };

    template <class GeneratorType>
    value_t sample(GeneratorType& gen) const {
        std::normal_distribution<value_t> dist(mean(), var());
        return dist(gen);
    }

    dist_value_t pdf(value_t x) const {
        return std::exp(- 0.5 * std::pow(x - mean(), 2) / var()) / (std::sqrt(var() * 2 * M_PI));
    }

    dist_value_t log_pdf(value_t x) const {
        return (-0.5 * std::pow(x - mean(), 2) / var()) - 0.5 * (std::log(var()) + std::log(2) + std::log(M_PI));
    }

    value_t mean() const { return static_cast<value_t>(mean_);}
    value_t var() const { return static_cast<value_t>(var_);}

private:
    mean_type mean_;
    var_type var_;
};

} // namespace dist
} // namespace ppl
