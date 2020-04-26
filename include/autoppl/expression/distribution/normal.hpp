#pragma once
#include <cassert>
#include <random>
#include <autoppl/util/var_expr_traits.hpp>
#include <autoppl/util/dist_expr_traits.hpp>

namespace ppl {
namespace expr {

template <typename mean_type, typename stddev_type>
struct Normal : util::DistExpr<Normal<mean_type, stddev_type>>
{
    static_assert(util::assert_is_var_expr_v<mean_type>);
    static_assert(util::assert_is_var_expr_v<stddev_type>);

    using value_t = util::cont_param_t;
    using dist_value_t = double;

    Normal(mean_type mean, stddev_type stddev)
        : mean_{mean}, stddev_{stddev} 
    {}

    template <class GeneratorType>
    value_t sample(GeneratorType& gen) const {
        std::normal_distribution dist(mean(), stddev());
        return dist(gen);
    }

    dist_value_t pdf(value_t x) const 
    {
        dist_value_t z_score = (x - mean()) / stddev();
        return std::exp(- 0.5 * z_score * z_score) / (stddev() * std::sqrt(2 * M_PI));
    }

    dist_value_t log_pdf(value_t x) const 
    {
        dist_value_t z_score = (x - mean()) / stddev();
        return -0.5 * ((z_score * z_score) + std::log(stddev() * stddev() * 2 * M_PI));
    }

    auto mean() const { return mean_.get_value(0);}
    auto stddev() const { return stddev_.get_value(0);}
    value_t min() const { return std::numeric_limits<value_t>::lowest(); }
    value_t max() const { return std::numeric_limits<value_t>::max(); }

private:
    mean_type mean_;
    stddev_type stddev_;
};

} // namespace expr
} // namespace ppl
