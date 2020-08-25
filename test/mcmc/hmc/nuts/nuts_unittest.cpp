#include "gtest/gtest.h" 
#include <array>
#include <fastad>
#include <autoppl/expression/model/bar_eq.hpp>
#include <autoppl/expression/model/glue.hpp>
#include <autoppl/expression/constraint/pos_def.hpp>
#include <autoppl/expression/variable/data.hpp>
#include <autoppl/expression/variable/param.hpp>
#include <autoppl/expression/variable/constant.hpp>
#include <autoppl/expression/variable/binary.hpp>
#include <autoppl/expression/variable/dot.hpp>
#include <autoppl/expression/distribution/bernoulli.hpp>
#include <autoppl/expression/distribution/uniform.hpp>
#include <autoppl/expression/distribution/normal.hpp>
#include <autoppl/expression/distribution/wishart.hpp>
#include <autoppl/expression/program/program.hpp>
#include <autoppl/expression/op_overloads.hpp>
#include <autoppl/mcmc/hmc/nuts/nuts.hpp>
#include <testutil/sample_tools.hpp>

namespace ppl {

struct nuts_tools_fixture : ::testing::Test
{
protected:
    mcmc::MomentumHandler<unit_var> m_handler;

    template <class VecType>
    double sample_average(const VecType& v)
    {
        return std::accumulate(v.data(), v.data() + v.size(), 0.)/v.size();
    }
};

TEST_F(nuts_tools_fixture, check_entropy_1d)
{
    using namespace mcmc;
    constexpr size_t n_params = 1;
    constexpr size_t n_vecs = 4;
    bool actual = false;

    Eigen::MatrixXd mat(n_params, n_vecs);
    auto theta_plus = mat.col(0);
    auto theta_minus = mat.col(1);
    auto rho_plus = mat.col(2);
    auto rho_minus = mat.col(3);
    
    theta_plus[0] = 1.;
    theta_minus[0] = 0.;
    rho_plus[0] = -1.;
    rho_minus[0] = 3.;

    // false test - cond 1 fails
    actual = check_entropy(theta_plus - theta_minus,
                           rho_plus, rho_minus);
    EXPECT_FALSE(actual);

    // false test - cond 2 fails
    rho_plus[0] = 1.;
    rho_minus[0] = -1.;
    actual = check_entropy(theta_plus - theta_minus,
                           rho_plus, rho_minus);
    EXPECT_FALSE(actual);
    
    // true test
    rho_plus[0] = 1.;
    rho_minus[0] = 3.; // reset to original
    actual = check_entropy(theta_plus - theta_minus,
                           rho_plus, rho_minus);
    EXPECT_TRUE(actual);
}


TEST_F(nuts_tools_fixture, check_entropy_3d)
{
    using namespace mcmc;
    constexpr size_t n_params = 3;
    constexpr size_t n_vecs = 4;
    bool actual = false;

    Eigen::MatrixXd mat(n_params, n_vecs);
    auto theta_plus = mat.col(0);
    auto theta_minus = mat.col(1);
    auto rho_plus = mat.col(2);
    auto rho_minus = mat.col(3);
    
    theta_plus[0] = 1.;
    theta_plus[1] = 2.;
    theta_plus[2] = 3.;
    theta_minus[0] = 0.;
    theta_minus[1] = -1.;
    theta_minus[2] = 0.5;
    rho_plus[0] = 1.;
    rho_plus[1] = 0.;
    rho_plus[2] = -2.;
    rho_minus[0] = 3.;
    rho_minus[1] = -4.;
    rho_minus[2] = 1.;

    // false test - cond 1 fails
    actual = check_entropy(theta_plus - theta_minus,
                           rho_plus, rho_minus);
    EXPECT_FALSE(actual);

    // false test - cond 2 fails
    rho_plus[2] = 2.;
    actual = check_entropy(theta_plus - theta_minus,
                           rho_plus, rho_minus);
    EXPECT_FALSE(actual);

    // true test
    rho_plus[2] = 5;    // makes dot == 0
    rho_minus[2] = 4;   // makes dot == 0
    actual = check_entropy(theta_plus - theta_minus,
                           rho_plus, rho_minus);
    EXPECT_TRUE(actual);
}

struct nuts_fixture : nuts_tools_fixture
{
protected:
    size_t n_samples = 5000;
    using value_t = double;
    using p_scl_t = ppl::Param<value_t>;
    using p_vec_t = ppl::Param<value_t, ppl::vec>;
    using d_vec_t = ppl::Data<value_t, ppl::vec>;

    p_scl_t w, b;
    d_vec_t x, y, q, r;

    NUTSConfig<> config;

    nuts_fixture()
        : w{}
        , b{}
        , x(6)
        , y(6)
        , q(6)
        , r(6)
    {
        x.get() << 2.5, 3, 3.5, 4, 4.5, 5.;
        y.get() << 3.5, 4, 4.5, 5, 5.5, 6.;
        q.get() << 2.4, 3.1, 3.6, 4, 4.5, 5.;
        r.get() << 3.5, 4, 4.4, 5.01, 5.46, 6.1;

        config.samples = n_samples;
        config.warmup = n_samples;
        config.seed = 0;
    }

    template <class MatType, class F>
    auto inv_transform(const Eigen::MatrixBase<MatType>& s,
                       size_t cols, 
                       F f)
    {
        Eigen::MatrixXd out(s.rows(), cols);
        for (int i = 0; i < s.rows(); ++i) {
            out.row(i) = f(s.row(i));
        }
        return out;
    }
};

TEST_F(nuts_fixture, nuts_std_normal)
{
    auto model = (
        w |= normal(0., 1.)
    );
    
    auto out = nuts(model, config);

    plot_hist(out.cont_samples.col(0));
    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 0., 0.05);
}

TEST_F(nuts_fixture, nuts_uniform)
{
    auto model = (
        w |= uniform(2., 3.)
    );

    auto out = nuts(model, config);

    plot_hist(out.cont_samples.col(0), 0.1);
    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 2.5, 0.02);
}

TEST_F(nuts_fixture, nuts_sample_unif_normal_posterior_stddev)
{
    Data<double> x(3.14);
    auto model = (
        w |= uniform(0.1, 5.),
        x |= normal(0., w)
    );
    auto out = nuts(model, config);
    plot_hist(out.cont_samples.col(0), 0.2);
    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 3.27226, 0.05);
}

TEST_F(nuts_fixture, nuts_sample_normal_stddev)
{
    Data<double> x(3.);
    auto model = (
        w |= normal(0., 2.),
        x |= normal(0., w * w)
    );
    auto out = nuts(model, config);
    plot_hist(out.cont_samples.col(0), 0.2); 
    // should be either gradually increasing then suddenly dropping to 0 or
    // suddenly jumping to 0 then gradually decreasing or
    // both of those (rare)
}

TEST_F(nuts_fixture, nuts_sample_unif_normal_posterior_mean)
{
    Data<double> x(3.);
    auto model = (
        w |= uniform(-20., 20.),
        x |= normal(w, 1.)
    );
    auto out = nuts(model, config);
    plot_hist(out.cont_samples.col(0)); 
    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 3.0, 0.03);
}

TEST_F(nuts_fixture, nuts_sample_regression_dist_weight) 
{
    auto model = (w |= normal(0., 2.),
                  y |= normal(x * w + 1., 0.5)
    );

    auto out = nuts(model, config);

    plot_hist(out.cont_samples.col(0), 0.1);
    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 1.0, 0.05);
}

TEST_F(nuts_fixture, nuts_sample_regression_dist_weight_bias) 
{
    auto model = (w |= normal(0., 2.),
                  b |= normal(0., 2.),
                  y |= normal(x * w + b, 0.5)
    );

    auto out = nuts(model, config);

    plot_hist(out.cont_samples.col(0), 0.1);
    plot_hist(out.cont_samples.col(1));
    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 1.0319, 0.06);
    EXPECT_NEAR(sample_average(out.cont_samples.col(1)), 0.8712, 0.08);
}

TEST_F(nuts_fixture, nuts_sample_regression_dist_uniform) {
    auto model = (w |= uniform(0., 2.),
                  b |= uniform(0., 2.),
                  y |= normal(x * w + b, 0.5)
    );

    auto out = nuts(model, config);

    plot_hist(out.cont_samples.col(0), 0.2, 0., 2.);
    plot_hist(out.cont_samples.col(1), 0.2, 0., 2.);

    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 1.0, 0.05);
    EXPECT_NEAR(sample_average(out.cont_samples.col(1)), 1.0, 0.05);
}

TEST_F(nuts_fixture, nuts_sample_regression_fuzzy_uniform) {
    auto model = (w |= uniform(0., 2.),
                  b |= uniform(0., 2.),
                  r |= normal(q * w + b, 0.5));

    auto out = nuts(model, config);

    plot_hist(out.cont_samples.col(0), 0.2, 0., 1.);
    plot_hist(out.cont_samples.col(1), 0.2, 0., 1.);

    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 1.0013, 0.05);
    EXPECT_NEAR(sample_average(out.cont_samples.col(1)), 0.9756, 0.05);
}

TEST_F(nuts_fixture, nuts_sample_regression_no_dot) {
    Eigen::VectorXd x_vec(3);
    x_vec(0) = 1.;
    x_vec(1) = -1.;
    x_vec(2) = 0.5;

    Eigen::VectorXd y_vec(3);
    y_vec(0) = 2.;
    y_vec(1) = -0.13;
    y_vec(2) = 1.32;

    DataView<double, vec> x(x_vec.data(), x_vec.size());
    DataView<double, vec> y(y_vec.data(), y_vec.size());

    auto model = (w |= uniform(0., 2.),
                  b |= uniform(0., 2.),
                  y |= normal(x*w + b, 0.5)
    );

    auto out = nuts(model, config);

    plot_hist(out.cont_samples.col(0), 0.2, 0., 2.);
    plot_hist(out.cont_samples.col(1), 0.2, 0., 2.);

    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 1.04, 0.05);
    EXPECT_NEAR(sample_average(out.cont_samples.col(1)), 0.89, 0.05);
}

TEST_F(nuts_fixture, nuts_sample_regression_dot) {
    Eigen::VectorXd x_mat(3);
    x_mat(0) = 1.;
    x_mat(1) = -1.;
    x_mat(2) = 0.5;

    Eigen::VectorXd y_vec(3);
    y_vec(0) = 2.;
    y_vec(1) = -0.13;
    y_vec(2) = 1.32;

    DataView<double, mat> x(x_mat.data(), x_mat.rows(), x_mat.cols());
    DataView<double, vec> y(y_vec.data(), y_vec.size());
    p_vec_t w(1);   // vector-shaped w instead of scl

    auto model = (w |= uniform(0., 2.),
                  b |= uniform(0., 2.),
                  y |= normal(ppl::dot(x, w) + b, 0.5)
    );

    auto out = nuts(model, config);

    plot_hist(out.cont_samples.col(0), 0.2, 0., 2.);
    plot_hist(out.cont_samples.col(1), 0.2, 0., 2.);

    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 1.0407, 0.05);
    EXPECT_NEAR(sample_average(out.cont_samples.col(1)), 0.8909, 0.05);
}

TEST_F(nuts_fixture, nuts_coin_flip) {
    std::vector<int> x_data({0, 1, 1});
    DataView<int, vec> x(x_data.data(), x_data.size());    

    auto model = (w |= uniform(0., 1.),
                  x |= bernoulli(w)
    );

    auto out = nuts(model, config);

    plot_hist(out.cont_samples.col(0), 0.1, 0., 1.);
    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 0.6, 0.02);
}

TEST_F(nuts_fixture, nuts_mean_vec_stddev_vec) {
    d_vec_t x(2);
    d_vec_t y(2);
    x.get() << 2.5, 3;
    y.get() << 3.5, 4;

    p_vec_t s(y.size());

    auto model = (w |= uniform(0., 2.),
                  b |= uniform(0., 2.),
                  s |= uniform(0.5, 5.),
                  y |= normal(x * w + b, s)
    );

    auto out = nuts(model, config);

    plot_hist(out.cont_samples.col(0), 0.2, 0., 2.);
    plot_hist(out.cont_samples.col(1), 0.2, 0., 2.);
    plot_hist(out.cont_samples.col(2), 0.25, 0.5, 5.);
    plot_hist(out.cont_samples.col(3), 0.25, 0.5, 5.);

    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 1.0, 0.25);
    EXPECT_NEAR(sample_average(out.cont_samples.col(1)), 1.0, 0.25);
    EXPECT_NEAR(sample_average(out.cont_samples.col(2)), 2.23439659, 0.25);
    EXPECT_NEAR(sample_average(out.cont_samples.col(3)), 2.30538608, 0.25);
}

TEST_F(nuts_fixture, nuts_wishart_cov) {
    d_vec_t y(2);
    y.get() << 1., -1.;
    Eigen::MatrixXd V(2,2);
    V << 2, 1, 1, 2;
    Param Sigma = make_param<double, mat>(2, pos_def());
    auto model = (Sigma |= wishart(V, 2.),
                  y |= normal(0, Sigma)
    );

    auto out = nuts(model, config);

    plot_hist(out.cont_samples.col(0));
    EXPECT_NEAR(out.cont_samples.col(0).mean(), 3.0, 0.2);
    EXPECT_NEAR(out.cont_samples.col(1).mean(), 0.12, 0.08);
    EXPECT_NEAR(out.cont_samples.col(2).mean(), 0.12, 0.08);
    EXPECT_NEAR(out.cont_samples.col(3).mean(), 3.0, 0.2);
}

} // namespace ppl
