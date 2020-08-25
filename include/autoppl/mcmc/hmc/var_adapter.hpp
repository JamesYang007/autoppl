#pragma once
#include <autoppl/math/welford.hpp>

namespace ppl {

/**
 * Type tags to indicate what kind of variance adapter.
 */
struct unit_var {};
struct diag_var {};
struct dense_var {};

/**
 * Configuration for variance adapter.
 * They will only be meaningful when policy is either diag_var or dense_var.
 */
struct VarConfig
{
    size_t init_buffer = 75;
    size_t term_buffer = 50;
    size_t window_base = 25;
};

namespace mcmc {

/**
 * Variance adapter.
 * @tparam  VarPolicy   one of unit_var, diag_var, dense_var
 */
template <class VarPolicy>
struct VarAdapter {};

/**
 * Unit variance with no adaptation.
 */
template <>
struct VarAdapter<unit_var>
{
    // For consistent API
    VarAdapter(size_t,
               size_t,
               size_t,
               size_t,
               size_t)
    {}
};

/**
 * Diagonal precision matrix M is estimated for momentum covariance matrix.
 * M inverse is estimated as sample variance and is regularized towards identity.
 *
 * Follows STAN guide: https://mc-stan.org/docs/2_18/reference-manual/hmc-algorithm-parameters.html
 * STAN implementation: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp
 */
template <>
struct VarAdapter<diag_var>
{
    VarAdapter(size_t n_params,
               size_t warmup,
               size_t init_buffer,
               size_t term_buffer,
               size_t window_base)
        : var_estimator_(n_params)
        , warmup_{warmup}
        , counter_{0}
        , window_begin_{init_buffer}
        , window_end_{warmup - term_buffer}
        , init_buffer_{init_buffer}
        , term_buffer_{term_buffer}
        , window_base_{window_base}
    {
        // constructor guarantees that at least 1 window computed
        
        // if warmup less than 20, just do 1 window
        if (warmup_ <= 20) {
            init_buffer_ = 0;
            term_buffer_ = 0;
            window_base_ = warmup_;
        }

        // else if warmup less than init + 1 window + term,
        // change to 15% init, 75% 1 window, 10% term (see STAN)
        else if (warmup_ < init_buffer_ + term_buffer_ + window_base_) {
            init_buffer_ = 0.15 * warmup_;
            term_buffer_ = 0.10 * warmup_;
            window_base_ = warmup_ - init_buffer_ - term_buffer_;
        }

        window_begin_ = init_buffer_; 
        window_end_ = window_begin_ + window_base_;
        size_t next_window_end = window_end_ + 2 * window_base_;

        // if next window ends lies inside term buffer, 
        // just make current window extend until term buffer
        if (next_window_end > warmup_ - term_buffer_) {
            window_end_ = warmup_ - term_buffer_;
        }
    }

    // If in init buffer or term buffer, don't adapt variance
    // otherwise, adapt variance if within window.
    // If reached end of current window, update variance, reset estimator, get new window.
    template <class MatType1, class MatType2>
    bool adapt(const Eigen::MatrixBase<MatType1>& x, 
               Eigen::MatrixBase<MatType2>& var) 
    {
        // if counter is not at the end of all windows
        if (counter_ >= init_buffer_ &&
            counter_ < warmup_ - term_buffer_) {
            var_estimator_.update(x);
        }

        // if currently at the end of the window,
        // get updated variance and reset estimator
        if (counter_ == window_end_ - 1) {
            auto&& v = var_estimator_.get_variance();
            double n = var_estimator_.get_n_samples();
            // regularized sample variance (see STAN)
            var.array() = ( (n / ((n + 5.0) * (n - 1.))) * v.array() + 
                            1e-3 * (5.0 / (n + 5.0)) );
            var_estimator_.reset();
            shift_window();
            ++counter_;
            return true;
        }

        // init or term buffer => no adapt variance
        static_cast<void>(x); 
        static_cast<void>(var); 
        ++counter_;
        return false;
    }

private:

    // invariant: at the beginning of the call,
    // next window is guaranteed to be fully before term buffer
    // OR current window reaches the end of the window
    void shift_window()
    {
        if (window_end_ == warmup_ - term_buffer_) { return; }

        size_t window_size = window_end_ - window_begin_;
        window_begin_ = window_end_; 
        window_end_ = window_begin_ + 2 * window_size;

        // optimization
        if (window_end_ == warmup_ - term_buffer_) { return; }

        size_t next_window_end = window_end_ + 4 * window_size;
        if (next_window_end > warmup_ - term_buffer_) {
            window_end_ = warmup_ - term_buffer_;
        }
    }

    math::WelfordVar var_estimator_;
    const size_t warmup_;
    size_t counter_;
    size_t window_begin_;
    size_t window_end_;
    size_t init_buffer_;
    size_t term_buffer_;
    size_t window_base_;
};

} // namespace mcmc
} // namespace ppl
