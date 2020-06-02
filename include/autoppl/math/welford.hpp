#pragma once
#include <type_traits>
#include <armadillo>

namespace ppl {
namespace math {

/*
 * Estimates sample variance for n-dimensional data using
 * Welford's online algorithm
 */
struct WelfordVar
{
    WelfordVar(size_t n_params)
        : m_{n_params, 2, arma::fill::zeros}
        , mean_{m_.col(0)}
        , m2n_{m_.col(1)}
    {}

    /*
     * Update sample mean and sample variance with new sample x.
     */
    template <class MatType>
    void update(const MatType& x)
    {
        ++n_;
        auto delta = x - mean_;
        mean_ += delta/static_cast<double>(n_);
        m2n_ += delta % (x - mean_);
    }

    /*
     * Populate v with sample variance vector.
     * If sample size is not greater than 1, v is zeroed out.
     */
    template <class MatType>
    void get_variance(MatType& v)
    { 
        if (n_ > 1) {
            v = m2n_/static_cast<double>(n_ - 1); 
        } else {
            v.zeros();
        }
    }

    size_t get_n_samples() const { return n_; }

    /*
     * Resets sample mean, sample variance, and number of samples to 0
     * Equivalent to constructing a new object of this type.
     */
    void reset()
    {
        m_.zeros();
        n_ = 0;
    }

private:
    arma::mat m_; 

    using col_t = std::decay_t<decltype(m_.col(0))>;
    col_t mean_;
    col_t m2n_;

    size_t n_ = 0;  // number of samples
};

} // namespace math
} // namespace ppl
