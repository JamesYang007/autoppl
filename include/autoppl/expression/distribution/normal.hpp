#pragma once
#include <cassert>
#include <random>
#include <autoppl/util/var_expr_traits.hpp>
#include <autoppl/util/dist_expr_traits.hpp>
#include <fastad>

namespace ppl {
namespace expr {

template <typename mean_type, typename stddev_type>
struct Normal : util::DistExpr<Normal<mean_type, stddev_type>>
{
    static_assert(util::assert_is_var_expr_v<mean_type>);
    static_assert(util::assert_is_var_expr_v<stddev_type>);

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
    
    auto mean(size_t index=0) const { return mean_.get_value(index);}
    auto stddev(size_t index=0) const { return stddev_.get_value(index);}

    dist_value_t log_pdf_no_constant(value_t x) const
    {
        dist_value_t z_score = (x - mean()) / stddev();
        return -0.5 * (z_score * z_score) - std::log(stddev());
    }

    /* 
     * Up to constant addition, returns ad expression of log pdf
     */
    template <class T, class VecRefType, class VecADVarType>
    auto ad_log_pdf(const ad::Var<T>& x,
                    const VecRefType& keys,
                    const VecADVarType& vars) const
    {
        auto ad_mean_expr = mean_.get_ad(keys, vars);
        auto ad_stddev_expr = stddev_.get_ad(keys, vars);
        return ((ad::constant(-0.5) * 
                ((x - ad_mean_expr) * (x - ad_mean_expr) / 
                    (ad_stddev_expr * ad_stddev_expr)))
                - ad::log(ad_stddev_expr)
               ); 
    }

    auto mean() const { return mean_.get_value();}
    auto stddev() const { return stddev_.get_value();}
    value_t min() const { return std::numeric_limits<value_t>::lowest(); }
    value_t max() const { return std::numeric_limits<value_t>::max(); }

private:
    mean_type mean_;  // TODO enforce that these are at least descended from a Param class.
    stddev_type stddev_;
};

} // namespace expr
} // namespace ppl
