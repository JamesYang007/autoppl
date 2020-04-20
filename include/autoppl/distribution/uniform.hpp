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

template <typename min_type, typename max_type>
struct Uniform : public DistExpr<Uniform<min_type, max_type>>
{
    using value_t = double;
    using param_value_t = std::common_type_t<
        typename var_traits<min_type>::value_t,
        typename var_traits<max_type>::value_t
            >;
    using dist_value_t = typename UniformBase::dist_value_t;

    Uniform(min_type min, max_type max)
        : min_{min}, max_{max} { assert(this -> min() < this -> max()); }

    // TODO: tag this class as "TriviallySamplable"?
    template <class GeneratorType>
    value_t sample(GeneratorType& gen) const
    {
        std::uniform_real_distribution dist(min(), max()); 
        return dist(gen);
    }

    dist_value_t pdf(value_t x) const
    { return UniformBase::pdf(x, min(), max()); }

    dist_value_t log_pdf(value_t x) const
    { return UniformBase::log_pdf(x, min(), max()); }

    param_value_t min() const { return static_cast<param_value_t>(min_); }
    param_value_t max() const { return static_cast<param_value_t>(max_); }

private:
    min_type min_;
    max_type max_;
};

} // namespace dist
} // namespace ppl
