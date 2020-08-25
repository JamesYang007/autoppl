#pragma once
#include <random>
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
    MomentumHandler(size_t=0) 
        : dist(0., 1.)
    {}

    /*
     * Sample from N(0, I)
     */
    template <class MatType
            , class GenType>
    void sample(Eigen::MatrixBase<MatType>& rho,
                GenType& gen) 
    { 
        rho = MatType::NullaryExpr(rho.rows(), 
                [&]() { return dist(gen); });
    }

    /**
     * Compute corresponding kinetic energy
     */
    template <class MatType>
    double kinetic(const Eigen::MatrixBase<MatType>& rho) const
    { return 0.5 * rho.squaredNorm(); }

    template <class MatType>
    const MatType& dkinetic_dr(const MatType& rho) const
    { return rho; }

private:
    std::normal_distribution<> dist;
};

/**
 * Diagonal variance with adaptation.
 */
template <>
struct MomentumHandler<diag_var>
{
    using adapter_policy_t = diag_var;
    using variance_t = Eigen::VectorXd;

    // initialize m inverse to be identity 
    MomentumHandler(size_t n_params)
        : dist(0., 1.)
        , m_inverse_(n_params)
    {
        m_inverse_.setOnes();
    }

    /**
     * Sample from N(0, M) where M inverse ~ sample variance matrix
     */
    template <class MatType
            , class GenType>
    void sample(Eigen::MatrixBase<MatType>& rho,
                GenType& gen) 
    { 
        rho = MatType::NullaryExpr(rho.rows(), 
                [&]() { return dist(gen); });
        rho.array() /= m_inverse_.array().sqrt();
    }

    /**
     * Compute corresponding kinetic energy
     */
    template <class MatType>
    double kinetic(const Eigen::MatrixBase<MatType>& rho) const
    { return 0.5 * rho.dot(dkinetic_dr(rho)); }

    template <class MatType>
    auto dkinetic_dr(const Eigen::MatrixBase<MatType>& rho) const
    { return (m_inverse_.array() * rho.array()).matrix(); }

    variance_t& get_m_inverse() { return m_inverse_; }
    const variance_t& get_m_inverse() const { return m_inverse_; }

private:
    std::normal_distribution<> dist;
    variance_t m_inverse_;
};

} // namespace mcmc
} // namespace ppl
