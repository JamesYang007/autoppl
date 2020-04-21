#pragma once
#include <cassert>
#include <random>
#include <autoppl/util/var_expr_traits.hpp>
#include <autoppl/util/dist_expr_traits.hpp>

namespace ppl {
namespace expr {

template <typename mean_type, typename stddev_type>
struct Normal
{
    static_assert(util::assert_is_var_expr_v<mean_type>);
    static_assert(util::assert_is_var_expr_v<stddev_type>);

    using value_t = util::cont_param_t;
    using param_value_t = std::common_type_t<
        typename util::var_expr_traits<mean_type>::value_t,
        typename util::var_expr_traits<stddev_type>::value_t
            >;
    using dist_value_t = double;

    Normal(mean_type mean, stddev_type stddev)
        : mean_{mean}, stddev_{stddev} {
            assert(this -> stddev() > 0);
        };

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

    param_value_t mean() const { return static_cast<param_value_t>(mean_);}
    param_value_t stddev() const { return static_cast<param_value_t>(stddev_);}

private:
    mean_type mean_;
    stddev_type stddev_;
};

} // namespace expr
} // namespace ppl
