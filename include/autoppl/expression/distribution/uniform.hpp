#pragma once
#include <cassert>
#include <random>
#include <autoppl/util/dist_expr_traits.hpp>
#include <autoppl/util/var_expr_traits.hpp>

namespace ppl {
namespace expr {

template <typename min_type, typename max_type>
struct Uniform : util::DistExpr<Uniform<min_type, max_type>>
{
    static_assert(util::assert_is_var_expr_v<min_type>);
    static_assert(util::assert_is_var_expr_v<max_type>);

    using value_t = util::cont_param_t;
    using dist_value_t = double;

    Uniform(min_type min, max_type max)
        : min_{min}, max_{max} 
    {}

    // TODO: tag this class as "TriviallySamplable"?
    template <class GeneratorType>
    value_t sample(GeneratorType& gen) const
    {
        std::uniform_real_distribution dist(min(), max()); 
        return dist(gen);
    }

    dist_value_t pdf(value_t x) const
    {
        return (min() < x && x < max()) ? 1. / (max() - min()) : 0;
    }

    dist_value_t log_pdf(value_t x) const
    {
        return (min() < x && x < max()) ? 
            -std::log(max() - min()) : 
            std::numeric_limits<dist_value_t>::lowest();
    }

    value_t min() const { return min_.get_value(); }
    value_t max() const { return max_.get_value(); }

private:
    min_type min_;
    max_type max_;
};

} // namespace expr
} // namespace ppl
