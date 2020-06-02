#pragma once
#include <autoppl/mcmc/hmc/var_adapter.hpp>

namespace ppl {
namespace mcmc {

/**
 * Adapts momentum variance and handles everything related to momentum.
 */
template <class AdapterPolicy>
struct MomentumHandler;

/**
 * Unit variance with no adaptation.
 */
template <>
struct MomentumHandler<unit_var>
{
    using adapter_policy_t = unit_var;

    // Constructor takes in size_t for consistent API with other specializations.
    MomentumHandler(size_t=0) {}

    /*
     * Sample from N(0, I)
     */
    template <class MatType>
    void sample(MatType& rho) const
    { rho.randn(); }

    /**
     * Compute corresponding kinetic energy
     */
    template <class MatType>
    double kinetic(const MatType& rho) const
    { return 0.5 * arma::dot(rho, rho); }

    template <class MatType>
    const MatType& dkinetic_dr(const MatType& rho) const
    { return rho; }

};

/**
 * Diagonal variance with adaptation.
 */
template <>
struct MomentumHandler<diag_var>
{
    using adapter_policy_t = diag_var;
    using variance_t = arma::vec;

    // initialize m inverse to be identity 
    MomentumHandler(size_t n_params)
        : m_inverse_(n_params, arma::fill::ones)
    {}

    /**
     * Sample from N(0, M) where M inverse ~ sample variance matrix
     */
    template <class MatType>
    void sample(MatType& rho) const
    { 
        rho.randn(); 
        rho /= arma::sqrt(m_inverse_);
    }

    /**
     * Compute corresponding kinetic energy
     */
    template <class MatType>
    double kinetic(const MatType& rho) const
    { return 0.5 * arma::dot(rho, m_inverse_ % rho); }

    template <class MatType>
    arma::vec dkinetic_dr(const MatType& rho) const
    { return m_inverse_ % rho; }

    variance_t& get_m_inverse() { return m_inverse_; }

private:
    variance_t m_inverse_;
};

} // namespace mcmc
} // namespace ppl
