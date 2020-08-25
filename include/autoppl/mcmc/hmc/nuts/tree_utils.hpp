#pragma once
#include <optional>
#include <functional>
#include <type_traits>

namespace ppl {
namespace mcmc {

/**
 * Struct to pack input to calling build_tree.
 */
template <class ADExprType
        , class SubviewType
        >
struct TreeInput
{
    using ad_expr_t = ADExprType;
    using subview_t = SubviewType;

    using ad_expr_ref_t = std::reference_wrapper<ad_expr_t>;
    using subview_ref_t = std::reference_wrapper<subview_t>;

    TreeInput(ad_expr_t& ad_expr,
              subview_t& theta,
              subview_t& theta_adj,
              subview_t& tp_adj,
              subview_t& theta_prime,
              subview_t& p_most,
              subview_t& p_beg, 
              subview_t& p_end, 
              subview_t& p_beg_scaled, 
              subview_t& p_end_scaled, 
              subview_t& rho,
              size_t& n_leapfrog,
              double& log_sum_weight,
              double& sum_metro_prob,
              int8_t v,
              double epsilon,
              double ham
               )
        : ad_expr_ref{ad_expr} 
        , theta_ref{theta}
        , theta_adj_ref{theta_adj}
        , tp_adj_ref{tp_adj}
        , theta_prime_ref{theta_prime}
        , p_most_ref{p_most}
        , p_beg_ref{p_beg}
        , p_end_ref{p_end}
        , p_beg_scaled_ref{p_beg_scaled}
        , p_end_scaled_ref{p_end_scaled}
        , rho_ref{rho}
        , n_leapfrog_ref{n_leapfrog}
        , log_sum_weight_ref{log_sum_weight}
        , sum_metro_prob_ref{sum_metro_prob}
        , v{v}
        , epsilon{epsilon}
        , ham{ham}
    {}

    ad_expr_ref_t ad_expr_ref;
    subview_ref_t theta_ref;
    subview_ref_t theta_adj_ref;
    subview_ref_t tp_adj_ref;
    subview_ref_t theta_prime_ref;
    subview_ref_t p_most_ref;   // either forward/backward-most momentum
    subview_ref_t p_beg_ref;    // begin new subtree (in the direction of v)
    subview_ref_t p_end_ref;    // end new subtree (in the direction of v)
    subview_ref_t p_beg_scaled_ref;
    subview_ref_t p_end_scaled_ref;
    subview_ref_t rho_ref;
    std::reference_wrapper<size_t> n_leapfrog_ref;
    std::reference_wrapper<double> log_sum_weight_ref;
    std::reference_wrapper<double> sum_metro_prob_ref;
    const int8_t v;
    const double epsilon;
    const double ham;
};

/**
 * Struct to pack output from calling build_tree.
 */
struct TreeOutput
{
    TreeOutput(bool _valid=true, double _potential=0.)
        : valid{_valid}, potential{_potential}
    {}

    bool valid;
    double potential;
};

} // namespace mcmc
} // namespace ppl
