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
    using base_t = util::DistExpr<Uniform<min_type, max_type>>; 
    using dist_value_t = typename base_t::dist_value_t;
    using base_t::pdf;
    using base_t::log_pdf;

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

    dist_value_t pdf(value_t x, size_t index=0) const
    {
        return (min(index) < x && x < max(index)) ? 1. / (max(index) - min(index)) : 0;
    }

    dist_value_t log_pdf(value_t x, size_t index=0) const
    {
        return (min(index) < x && x < max(index)) ? 
            -std::log(max(index) - min(index)) : 
            std::numeric_limits<dist_value_t>::lowest();
    }

    value_t min(size_t index=0) const { return min_.get_value(index); }
    value_t max(size_t index=0) const { return max_.get_value(index); }

private:
    min_type min_;  // TODO enforce that these are at least descended from a Param class.
    max_type max_;
};

} // namespace expr
} // namespace ppl
