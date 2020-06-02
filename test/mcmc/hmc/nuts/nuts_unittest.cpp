#include "gtest/gtest.h" 
#include <array>
#include <autoppl/expr_builder.hpp>
#include <autoppl/mcmc/hmc/nuts/nuts.hpp>
#include <testutil/sample_tools.hpp>
#include <fastad>

namespace ppl {

struct nuts_tools_fixture : ::testing::Test
{
protected:
    mcmc::MomentumHandler<unit_var> m_handler;

    template <class VecType>
    double sample_average(const VecType& v)
    {
        return std::accumulate(v.begin(), v.end(), 0.)/v.size();
    }
};

TEST_F(nuts_tools_fixture, check_entropy_1d)
{
    using namespace mcmc;
    constexpr size_t n_params = 1;
    constexpr size_t n_vecs = 4;
    bool actual = false;

    arma::mat mat(n_params, n_vecs);
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

    arma::mat mat(n_params, n_vecs);
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

/*
 * Fixture just to test for this build_tree function
 */
struct nuts_build_tree_fixture : nuts_tools_fixture
{
protected:
    using ad_vec_t = std::vector<ad::Var<double>>;
    size_t n_params = 3;
    ad_vec_t ad_vars;
    arma::mat data;

    using subview_t = std::decay_t<decltype(data.col(0))>;
    subview_t theta;
    subview_t theta_adj;
    subview_t rho;
    subview_t opt_theta;
    subview_t opt_rho;
    subview_t theta_prime;

    int8_t v = 1;
    double epsilon = 2.;
    double ham = 0.;

    mcmc::TreeOutput output;

    std::uniform_real_distribution<double> unif_sampler;
    std::mt19937 gen;

    mcmc::MomentumHandler<unit_var> m_handler;

    nuts_build_tree_fixture()
        : ad_vars(3)
        , data(n_params, 6)
        , theta(data.col(0))
        , theta_adj(data.col(1))
        , rho(data.col(2))
        , opt_theta(data.col(3))
        , opt_rho(data.col(4))
        , theta_prime(data.col(5))
        , output()
        , unif_sampler(0., 1.)
    {
        // bind theta and theta_adj to be the value/adj storage
        mcmc::ad_bind_storage(ad_vars, theta, theta_adj);

        // initialize theta and rho
        theta[0] = 0.; theta[1] = 0.; theta[2] = 0.;
        rho[0] = -1.; rho[1] = 0.; rho[2] = 1.;

        // theta adjoint MUST be set
        theta_adj[0] = 0.; theta_adj[1] = 0.; theta_adj[2] = 0.;
    }
};

TEST_F(nuts_build_tree_fixture, find_reasonable_log_epsilon)
{
    auto ad_expr = ad::constant(-0.5) * 
                   (ad_vars[0] * ad_vars[0] + 
                    ad_vars[1] * ad_vars[1] +
                    ad_vars[2] * ad_vars[2]
                   ) ;
    double eps = mcmc::find_reasonable_epsilon<3>(1., ad_expr, theta, theta_adj, m_handler);
    static_cast<void>(eps);
}

struct nuts_fixture : nuts_tools_fixture
{
protected:
    size_t n_samples = 5000;
    std::vector<double> w_storage, b_storage;
    Param<double> w, b;
    ppl::Data<double> x {2.5, 3, 3.5, 4, 4.5, 5.};
    ppl::Data<double> y {3.5, 4, 4.5, 5, 5.5, 6.};
    ppl::Data<double> q{2.4, 3.1, 3.6, 4, 4.5, 5.};
    ppl::Data<double> r{3.5, 4, 4.4, 5.01, 5.46, 6.1};
    NUTSConfig<> config;

    nuts_fixture()
        : w_storage(n_samples, 0.)
        , b_storage(n_samples, 0.)
        , w{w_storage.data()}
        , b{b_storage.data()}
    {
        config.n_samples = n_samples;
        config.warmup = n_samples;
        config.seed = 1;
    }

    void reconfigure(size_t n)
    {
        w_storage.resize(n);
        b_storage.resize(n);
        w.set_storage(w_storage.data());
        b.set_storage(b_storage.data());
    }
};

TEST_F(nuts_fixture, nuts_std_normal)
{
    auto model = (
        w |= normal(0., 1.)
    );

    nuts(model, config);

    plot_hist(w_storage);
    EXPECT_NEAR(sample_average(w_storage), 0., 0.1);
}

TEST_F(nuts_fixture, nuts_uniform)
{
    auto model = (
        w |= uniform(0., 1.)
    );

    nuts(model, config);

    plot_hist(w_storage, 0.1);
    EXPECT_NEAR(sample_average(w_storage), 0.5, 0.1);
}

TEST_F(nuts_fixture, nuts_sample_unif_normal_posterior_stddev)
{
    Data<double> x(3.14);
    auto model = (
        w |= uniform(0.1, 5.),
        x |= normal(0., w)
    );
    nuts(model, config);
    plot_hist(w_storage, 0.2);
    EXPECT_NEAR(sample_average(w_storage), 3.27226, 0.1);
}

TEST_F(nuts_fixture, nuts_sample_normal_stddev)
{
    Data<double> x(3.);
    auto model = (
        w |= normal(0., 2.),
        x |= normal(0., w * w)
    );
    nuts(model, config);
    plot_hist(w_storage, 0.2); 
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
    nuts(model, config);
    plot_hist(w_storage); 
    EXPECT_NEAR(sample_average(w_storage), 3.0, 0.1);
}

TEST_F(nuts_fixture, nuts_sample_regression_dist_weight) 
{
    auto model = (w |= normal(0., 2.),
                  y |= normal(x * w + 1., 0.5)
    );

    nuts(model, config);

    plot_hist(w_storage, 0.1);
    EXPECT_NEAR(sample_average(w_storage), 1.0, 0.1);
}

TEST_F(nuts_fixture, nuts_sample_regression_dist_weight_bias) 
{
    auto model = (b |= normal(0., 2.),
                  w |= normal(0., 2.),
                  y |= normal(x * w + b, 0.5)
    );

    nuts(model, config);

    plot_hist(w_storage, 0.1);
    plot_hist(b_storage);
    EXPECT_NEAR(sample_average(w_storage), 1.0, 0.1);
    EXPECT_NEAR(sample_average(b_storage), 1.0, 0.3);
}

TEST_F(nuts_fixture, nuts_sample_regression_dist_uniform) {
    auto model = (w |= uniform(0., 2.),
                  b |= uniform(0., 2.),
                  y |= normal(x * w + b, 0.5)
    );

    nuts(model, config);

    plot_hist(w_storage, 0.2, 0., 2.);
    plot_hist(b_storage, 0.2, 0., 2.);

    EXPECT_NEAR(sample_average(w_storage), 1.0, 0.1);
    EXPECT_NEAR(sample_average(b_storage), 1.0, 0.1);
}

TEST_F(nuts_fixture, nuts_sample_regression_fuzzy_uniform) {
    auto model = (w |= uniform(0., 2.),
                  b |= uniform(0., 2.),
                  r |= normal(q * w + b, 0.5));

    nuts(model, config);

    plot_hist(w_storage, 0.2, 0., 1.);
    plot_hist(b_storage, 0.2, 0., 1.);

    EXPECT_NEAR(sample_average(w_storage), 1.0, 0.1);
    EXPECT_NEAR(sample_average(b_storage), 0.95, 0.1);
}

} // namespace ppl
