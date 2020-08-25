#pragma once
#include <type_traits>
#include <Eigen/Dense>

namespace ppl {
namespace math {

/**
 * Estimates sample variance for n-dimensional data using
 * Welford's online algorithm
 */
struct WelfordVar
{
    WelfordVar(size_t n_params)
        : m_{n_params, 2}
        , mean_{m_.col(0)}
        , m2n_{m_.col(1)}
    { m_.setZero(); }

    /*
     * Update sample mean and sample variance with new sample x.
     */
    template <class MatType>
    void update(const MatType& x)
    {
        ++n_;
        auto delta = x - mean_;
        mean_ += (1./static_cast<double>(n_)) * delta;
        m2n_ += (delta.array() * (x - mean_).array()).matrix();
    }

    const auto& get_variance() const { return m2n_; }
    size_t get_n_samples() const { return n_; }

    /**
     * Resets sample mean, sample variance, and number of samples to 0
     * Equivalent to constructing a new object of this type.
     */
    void reset()
    {
        m_.setZero();
        n_ = 0;
    }

private:
    Eigen::MatrixXd m_; 

    using col_t = std::decay_t<decltype(m_.col(0))>;
    col_t mean_;
    col_t m2n_;

    size_t n_ = 0;  // number of samples
};

} // namespace math
} // namespace ppl
