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

template <typename mean_type, typename stddev_type>
struct Normal : public DistExpr<Normal<mean_type, stddev_type>>
{
    using value_t = double;
    using param_value_t = std::common_type_t<
        typename var_traits<mean_type>::value_t,
        typename var_traits<stddev_type>::value_t
            >;
    using dist_value_t = typename NormalBase::dist_value_t;

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
    { return NormalBase::pdf(x, mean(), stddev()); }

    dist_value_t log_pdf(value_t x) const 
    { return NormalBase::log_pdf(x, mean(), stddev()); }

    param_value_t mean() const { return static_cast<param_value_t>(mean_);}
    param_value_t stddev() const { return static_cast<param_value_t>(stddev_);}

private:
    mean_type mean_;
    stddev_type stddev_;
};

} // namespace dist
} // namespace ppl
