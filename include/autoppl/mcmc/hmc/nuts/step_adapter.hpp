#pragma once
#include <cstddef>
#include <cmath>

namespace ppl {

/*
 * Constants that can be set by user and used by NUTSAdapter.
 */
struct StepConfig
{
    double delta = 0.8;
    double gamma = 0.05;
    double t0 = 10.;
    double kappa = 0.75;
};

namespace mcmc {

/*
 * Adaptive values in NUTS algorithm using dual-averaging.
 * Currently, only log_eps, log_eps_bar, and H_bar are adaptive.
 */
struct StepAdapter
{
    /*
     * Constructs a step adapter with the given log epsilon.
     * Initializes mu to log(10*_log_eps).
     */
    StepAdapter(double _log_eps)
    {
        init(_log_eps);
    }

    /*
     * Must be called before beginning to adapt.
     */
    void init(double _log_eps) 
    {
        log_eps = _log_eps;
        mu = mu_constant + log_eps;
    }

    /*
     * Adapts log_eps, log_eps_bar, and H_bar.
     */
    void adapt(double alpha_ratio)
    {
        ++counter;
        alpha_ratio = (alpha_ratio > 1) ? 1 : alpha_ratio;
        const double adapt_ratio = 1./(counter + step_config.t0);
        H_bar = (1 - adapt_ratio) * H_bar + 
                adapt_ratio * (step_config.delta - alpha_ratio);
        log_eps = mu - std::sqrt(counter)/step_config.gamma * H_bar;
        const double m_ratio = std::pow(counter, -step_config.kappa);
        log_eps_bar = m_ratio * log_eps +
                      (1 - m_ratio) * log_eps_bar;
    }

    /*
     * Reset all variables that change per iteration
     */
    void reset() 
    {
        counter = 0;
        log_eps_bar = 0.;
        H_bar = 0.;
    }

    size_t counter = 0;
    double log_eps = 0.;
    double log_eps_bar = 0.;
    double H_bar = 0.;
    double mu = 0.;
    const double mu_constant = std::log(10);
    StepConfig step_config;
};

} // namespace mcmc
} // namespace ppl
