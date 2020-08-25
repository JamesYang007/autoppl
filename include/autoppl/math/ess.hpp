#pragma once
#include <Eigen/Dense>
#include <autoppl/math/autocorrelation.hpp>

namespace ppl {
namespace math {
namespace details {

template <class T>
using vec_cref_t = std::vector<
    std::reference_wrapper<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> >;

} // namespace details

/**
 * Computes the effective sample size (ESS) for a given vector of samples (matrices).
 * Every element of the vector is a matrix of samples for each chain.
 * Every matrix contains the samples as rows, i.e.
 * every row is a sample of an p-dimensional vector, where p
 * is the number of columns of the matrix (number of parameters).
 *
 * The algorithm assumes that every sample matrix has the same dimensions.
 *
 * @tparam  T           underlying Eigen expression type
 * @param   samples     vector of samples
 *
 * @return  a vector of ESS for each component
 *          If number of samples is 1 or less, or there are 0 components,
 *          or number of chains is 0, return an empty vector.
 *          In either case, the dimension of the return vector is same
 *          as the number of components.
 */
template <class T>
inline auto ess(const details::vec_cref_t<T>& samples)
{
    using value_t = T;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;

    vec_t tau_hat;

    size_t M = samples.size();      // number of chains

    if (M == 0) return tau_hat;

    size_t dim = samples[0].get().cols();    // sample dimension
    size_t N = samples[0].get().rows();      // number of samples

    if (N <= 1 || dim == 0) return tau_hat;

    tau_hat = vec_t::Zero(dim);

    auto var = [N](const auto& mat) {
        return (mat.rowwise() - 
                mat.colwise().mean()).colwise()
                                     .squaredNorm() / (N-1);
    };

    // use N-1 scaling to compute variance
    // each col is the sample variance per chain
    mat_t sample_vars(dim, M);
    for (size_t i = 0; i < M; ++i) {
        sample_vars.col(i) = var(samples[i].get());
    }

    // column vector of average of sample variances
    vec_t W = sample_vars.rowwise().mean();

    // compute variance estimator
    vec_t var_est = static_cast<T>(N-1) / N * W;

    // if there is more than 1 chain, then update by N * B
    // where B is the between-chain variance
    if (M > 1) {
        mat_t sample_mean(dim, M);
        for (size_t i = 0; i < M; ++i) {
            sample_mean.col(i) = samples[i].get().colwise().mean();
        }
        var_est += var(sample_mean.transpose());
    }

    // compute autocorrelation vector for each component
    // every column vector (every component) is average AC over chains
    mat_t acov_mean(N, dim);
    acov_mean.setZero();

    for (size_t m = 1; m <= M; ++m) {
        mat_t next_acov = autocorrelation(samples[m-1].get());
        for (int j = 0; j < next_acov.cols(); ++j) {
            next_acov.col(j) *= sample_vars(j,m-1);
        }
        value_t m_inv = 1./m;
        acov_mean = m_inv * next_acov + (m-1) * m_inv * acov_mean;
    }
    
    // compute rho-hat at lag t for dimension d
    auto rho_hat = [&](size_t t, size_t d) {
        return 1. - (W(d) - acov_mean(t,d))/var_est(d);
    };

    // compute tau-hat directly to save memory
    for (size_t d = 0; d < dim; ++d) {
        
        // first two should not be corrected for positive and monotoneness
        value_t curr_rho_hat_even = rho_hat(0,d);
        value_t curr_p_hat = curr_rho_hat_even + rho_hat(1,d);  // current P_hat(t)
        value_t curr_min = curr_p_hat;                          // current min of P_hat(t)
        tau_hat(d) = curr_min;                                  // update with P_hat(0)

        // only estimate up to 3 samples before the end
        // and Geyer's positive condition holds
        size_t t = 2;
        for (; t < (N-3) && curr_p_hat > 0; t += 2) {
            curr_rho_hat_even = rho_hat(t,d);
            curr_p_hat = curr_rho_hat_even + rho_hat(t+1,d);

            // if positive condition holds, take the min 
            // of current P_hat(t) with the min of previous P_hat's
            // to create a monotone sequence and accumulate to tau_hat
            if (curr_p_hat >= 0) {
                curr_min = std::min(curr_min, curr_p_hat);
                tau_hat(d) += curr_min;
            }
        }

        // correct to improve estimate (see STAN's implementation)
        value_t correction = (curr_rho_hat_even > 0) ? 
                curr_rho_hat_even : rho_hat(t,d);

        tau_hat(d) *= 2.; // 2 * sum of adjusted P_hat(t)
        tau_hat(d) -= 1.; // -1 + 2 * sum of adjusted P_hat(t) 
        tau_hat(d) += correction;   
    }

    vec_t n_eff = N*M*(1./tau_hat.array()).min(std::log10(N)).matrix();

    return n_eff;
}

/**
 * Computes the effective sample size (ESS) for a given sample matrix.
 * This is an overload for when there is only chain and can supply
 * a single matrix instead.
 * See above overload for more details.
 *
 * @tparam  T           underlying Eigen expression type
 * @param   samples     sample matrix (1 chain)
 *
 * @return  a vector of ESS for each component
 */
template <class T>
inline auto ess(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& samples)
{
    details::vec_cref_t<T> v;
    v.emplace_back(samples);
    return ess(v);
}
    

} // namespace math
} // namespace ppl
