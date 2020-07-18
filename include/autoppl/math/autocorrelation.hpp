#pragma once
#include <armadillo>

namespace ppl {
namespace math {

inline size_t padded_length(size_t N)
{
    return std::pow(2, std::ceil(std::log(N)/std::log(2.)));
}

/**
 * Computes autocorrelation of x where each column of x
 * is a component of a process and hence each row is a time point.
 * More mathematically, x(i,...) is the process value at time i.
 *
 * For more detail, see: 
 * https://lingpipe-blog.com/2012/06/08/autocorrelation-fft-kiss-eigen/
 * https://github.com/stan-dev/math/blob/41e548e19da5675121c245b535d4019c8bbd754b/stan/math/prim/mat/fun/autocorrelation.hpp
 *
 * Currently, this functionality is only available for armadillo matrices.
 * @tparam  T   underlying value type (usually double)
 * @param   x   process matrix
 */
template <class T>
inline auto autocorrelation(const arma::Mat<T>& x)
{
    using complex_t = std::complex<T>;

    size_t n_rows = x.n_rows;
    size_t padded_len = 2*padded_length(n_rows);

    // create centered copy of x
    arma::Mat<T> x_mean = arma::mean(x, 0);
    arma::Mat<T> x_cent(x);
    x_cent.each_row([&](arma::rowvec& row) {
                row -= x_mean;
            });

    // FFT 
    arma::Mat<complex_t> freq = arma::fft(x_cent, padded_len);

    // compute complex-norm element-wise
    freq.for_each([](complex_t& elt) {
                elt = std::norm(elt);
            });

    // inverse FFT and trim to shape of x
    arma::Mat<complex_t> ifreq = arma::ifft(freq);
    auto ifreq_trim = ifreq.submat(0,0,arma::size(x));

    // get autocorrelation by normalizing by variance
    arma::Mat<T> autocorr = arma::real(ifreq_trim) / (n_rows * n_rows * 2.);
    autocorr.each_col([&](arma::vec& col) { col /= col(0); });

    return autocorr;
}

} // namespace math
} // namespace ppl
