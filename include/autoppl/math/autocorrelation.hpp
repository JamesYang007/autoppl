#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

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
 * @tparam  T   underlying value type (usually double)
 * @param   x   process matrix
 */
template <class T>
inline auto autocorrelation(const Eigen::MatrixBase<T>& x)
{
    using scalar_t = typename T::Scalar;
    using complex_t = std::complex<scalar_t>;

    size_t n_rows = x.rows();
    size_t padded_len = 2*padded_length(n_rows);

    Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic> out(x.rows(), x.cols());

    Eigen::FFT<scalar_t> fft;

    for (int i = 0; i < x.cols(); ++i) {

        // create centered copy of x
        Eigen::Matrix<scalar_t, Eigen::Dynamic, 1> x_cent(padded_len);
        x_cent.setZero();
        x_cent.head(n_rows) = x.col(i).array() - x.col(i).mean();

        // FFT 
        Eigen::Matrix<complex_t, Eigen::Dynamic, 1> freq(padded_len);
        fft.fwd(freq, x_cent);

        // compute complex-norm element-wise
        freq = freq.array().abs2();

        // inverse FFT and trim to shape of x
        Eigen::Matrix<complex_t, Eigen::Dynamic, 1> ifreq(padded_len);
        fft.inv(ifreq, freq);

        // get autocorrelation by normalizing by variance
        out.col(i) = ifreq.head(n_rows).real() / (n_rows * n_rows * 2.);
        out.col(i) /= out(0,i);

    }

    return out;
}

} // namespace math
} // namespace ppl
