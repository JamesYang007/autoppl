#pragma once
#include <armadillo>
#include <autoppl/math/autocorrelation.hpp>

namespace ppl {
namespace math {

/**
 * Computes the effective sample size (ESS) for a given sample cube.
 * Every slice of a cube is a matrix of samples for each chain.
 * Every matrix contains the samples as rows, i.e.
 * every row is a sample of an n-dimensional vector, where n
 * is the number of columns of the matrix.
 *
 * @tparam  T           underlying data type
 * @param   samples     sample cube
 *
 * @return  a vector of ESS for each component
 *          If number of samples is 1 or less, or there are 0 components,
 *          or number of chains is 0, return a vector of zeros.
 *          In either case, the dimension of the return vector is same
 *          as the number of components.
 */
template <class T>
inline arma::Col<T> ess(const arma::Cube<T>& samples)
{
    size_t dim = samples.n_cols;    // sample dimension
    size_t N = samples.n_rows;      // number of samples
    size_t M = samples.n_slices;    // number of chains

    arma::Col<T> tau_hat(dim, arma::fill::zeros);

    if (N <= 1 || M == 0 || dim == 0) return tau_hat;

    // use N-1 scaling to compute variance
    // each col is the sample variance per chain
    arma::Mat<T> sample_vars(dim, M);
    for (size_t i = 0; i < M; ++i) {
        sample_vars.col(i) = 
            arma::var(samples.slice(i), 0, 0).as_col(); 
    }

    // column vector of average of sample variances
    arma::Col<T> W = arma::mean(sample_vars, 1);

    // compute variance estimator
    arma::Col<T> var_est = static_cast<T>(N-1) / N * W;

    // if there is more than 1 chain, then update by N * B
    // where B is the between-chain variance
    arma::Mat<T> sample_mean = arma::mean(samples, 0);
    if (M > 1) var_est += arma::var(sample_mean, 0, 1);

    // compute autocorrelation vector for each component
    // every column vector (every component) is average AC over chains
    arma::Mat<T> acov_mean(N, dim, arma::fill::zeros);
    for (size_t m = 1; m <= M; ++m) {
        arma::Mat<T> next_acov = autocorrelation(samples.slice(m-1));
        for (size_t j = 0; j < next_acov.n_cols; ++j) {
            next_acov.col(j) *= sample_vars(j,m-1);
        }
        T m_inv = 1./m;
        acov_mean = m_inv * next_acov + (m-1) * m_inv * acov_mean;
    }
    
    // compute rho-hat at lag t for dimension d
    auto rho_hat = [&](size_t t, size_t d) {
        return 1. - (W(d) - acov_mean(t,d))/var_est(d);
    };

    // compute tau-hat directly to save memory
    for (size_t d = 0; d < dim; ++d) {
        
        // first two should not be corrected for positive and monotoneness
        T curr_rho_hat_even = rho_hat(0,d);
        T curr_p_hat = curr_rho_hat_even + rho_hat(1,d);  // current P_hat(t)
        T curr_min = curr_p_hat;                          // current min of P_hat(t)
        tau_hat(d) = curr_min;                            // update with P_hat(0)

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
        T correction = (curr_rho_hat_even > 0) ? 
            curr_rho_hat_even : rho_hat(t,d);

        tau_hat(d) *= 2.; // 2 * sum of adjusted P_hat(t)
        tau_hat(d) -= 1.; // -1 + 2 * sum of adjusted P_hat(t) 
        tau_hat(d) += correction;   
    }

    arma::Col<T> n_eff = 1./tau_hat;
    return N*M*arma::clamp(n_eff, n_eff.min(), std::log10(N));
}

/**
 * Computes the effective sample size (ESS) for a given sample matrix.
 * This is an overload for when there is only chain and can supply
 * a single matrix instead.
 * See above overload for more details.
 * Note that we take in by non-const lvalue reference to 
 * fit the API for armadillo in ensuring there is no copy of data.
 * However, this function does not modify samples.
 * It simply makes a cube viewing this matrix and delegates the call
 * to the overload above, which takes in a const reference.
 *
 * @tparam  T           underlying data type
 * @param   samples     sample matrix
 *
 * @return  a vector of ESS for each component
 */
template <class T>
inline arma::Col<T> ess(arma::Mat<T>& samples)
{
    size_t n_rows = samples.n_rows;
    size_t n_cols = samples.n_cols;
    size_t n_slices = 1;
    arma::Cube<T> cubed(samples.memptr(), n_rows, 
                        n_cols, n_slices,
                        false,
                        true);
    return ess(cubed);
}
    

} // namespace math
} // namespace ppl
