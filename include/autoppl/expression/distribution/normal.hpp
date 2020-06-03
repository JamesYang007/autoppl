#pragma once
#include <cmath>
#include <cassert>
#include <random>
#include <autoppl/util/var_expr_traits.hpp>
#include <autoppl/util/dist_expr_traits.hpp>
#include <fastad>

// MSVC does not seem to support M_PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ppl {
namespace expr {

#if __cplusplus <= 201703L
template <typename mean_type, typename stddev_type>
#else
template <util::var_expr mean_type, util::var_expr stddev_type>
#endif
struct Normal : util::DistExpr<Normal<mean_type, stddev_type>>
{

#if __cplusplus <= 201703L
    static_assert(util::assert_is_var_expr_v<mean_type>);
    static_assert(util::assert_is_var_expr_v<stddev_type>);
#endif

    using value_t = util::cont_param_t;
    using base_t = util::DistExpr<Normal<mean_type, stddev_type>>;
    using dist_value_t = typename base_t::dist_value_t;
    using base_t::pdf;
    using base_t::log_pdf;

    Normal(mean_type mean, stddev_type stddev)
        : mean_{mean}, stddev_{stddev} 
    {}

    template <class GeneratorType>
    value_t sample(GeneratorType& gen) const {
        std::normal_distribution dist(mean(), stddev());
        return dist(gen);
    }

    dist_value_t pdf(value_t x, size_t index=0) const 
    {
        dist_value_t z_score = (x - mean(index)) / stddev(index);
        return std::exp(- 0.5 * z_score * z_score) / (stddev(index) * std::sqrt(2 * M_PI));
    }

    dist_value_t log_pdf(value_t x, size_t index=0) const 
    {
        dist_value_t z_score = (x - mean(index)) / stddev(index);
        return -0.5 * ((z_score * z_score) + std::log(stddev(index) * stddev(index) * 2 * M_PI));
    }
    
    /** 
     * Up to constant addition, returns ad expression of log pdf
     */
    template <class ADVarType, class VecRefType, class VecADVarType>
    auto ad_log_pdf(const ADVarType& x,
                    const VecRefType& keys,
                    const VecADVarType& vars,
                    size_t idx = 0) const
    {
        auto&& ad_mean_expr = mean_.get_ad(keys, vars, idx);
        auto&& ad_stddev_expr = stddev_.get_ad(keys, vars, idx);
        return ad::if_else(
                ad_stddev_expr > ad::constant(0.),
                (ad::constant(-0.5) * 
                ad::pow<2>((x - ad_mean_expr) / ad_stddev_expr))
                - ad::log(ad_stddev_expr),
                ad::constant(std::numeric_limits<dist_value_t>::lowest())
               ); 
    }

    auto mean(size_t index=0) const { return mean_.get_value(index);}
    auto stddev(size_t index=0) const { return stddev_.get_value(index);}
    value_t min() const { return std::numeric_limits<value_t>::lowest(); }
    value_t max() const { return std::numeric_limits<value_t>::max(); }

private:
    mean_type mean_;  // TODO enforce that these are at least descended from a Param class.
    stddev_type stddev_;
};

} // namespace expr
} // namespace ppl
