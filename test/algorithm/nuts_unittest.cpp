#include "gtest/gtest.h" 
#include <array>
#include <autoppl/expr_builder.hpp>
#include <autoppl/algorithm/nuts.hpp>
#include <testutil/sample_tools.hpp>
#include <fastad>

namespace ppl {

struct nuts_tools_fixture : ::testing::Test
{
protected:
    template <class VecType>
    double sample_average(const VecType& v)
    {
        return std::accumulate(v.begin(), v.end(), 0.)/v.size();
    }
};

TEST_F(nuts_tools_fixture, check_entropy_1d)
{
    using namespace alg;
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
    actual = check_entropy(theta_plus, theta_minus,
                           rho_plus, rho_minus);
    EXPECT_FALSE(actual);

    // false test - cond 2 fails
    rho_plus[0] = 1.;
    rho_minus[0] = -1.;
    actual = check_entropy(theta_plus, theta_minus,
                           rho_plus, rho_minus);
    EXPECT_FALSE(actual);
    
    // true test
    rho_plus[0] = 1.;
    rho_minus[0] = 3.; // reset to original
    actual = check_entropy(theta_plus, theta_minus,
                           rho_plus, rho_minus);
    EXPECT_TRUE(actual);
}


TEST_F(nuts_tools_fixture, check_entropy_3d)
{
    using namespace alg;
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
    actual = check_entropy(theta_plus, theta_minus,
                           rho_plus, rho_minus);
    EXPECT_FALSE(actual);

    // false test - cond 2 fails
    rho_plus[2] = 2.;
    actual = check_entropy(theta_plus, theta_minus,
                           rho_plus, rho_minus);
    EXPECT_FALSE(actual);

    // true test
    rho_plus[2] = 5;    // makes dot == 0
    rho_minus[2] = 4;   // makes dot == 0
    actual = check_entropy(theta_plus, theta_minus,
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

    using subview_t = std::decay_t<decltype(data.unsafe_col(0))>;
    subview_t theta;
    subview_t theta_adj;
    subview_t rho;
    subview_t opt_theta;
    subview_t opt_rho;
    subview_t theta_prime;

    double log_u = -0.67;
    int8_t v = 1;
    double epsilon = 2.;
    double ham = 0.;

    using output_t = alg::TreeOutput<subview_t>;
    output_t output;

    nuts_build_tree_fixture()
        : ad_vars(3)
        , data(n_params, 6)
        , theta(data.unsafe_col(0))
        , theta_adj(data.unsafe_col(1))
        , rho(data.unsafe_col(2))
        , opt_theta(data.unsafe_col(3))
        , opt_rho(data.unsafe_col(4))
        , theta_prime(data.unsafe_col(5))
        , output(theta_prime)
    {
        // bind theta and theta_adj to be the value/adj storage
        alg::ad_bind_storage(ad_vars, theta, theta_adj);

        // initialize theta and rho
        theta[0] = 0.; theta[1] = 0.; theta[2] = 0.;
        rho[0] = -1.; rho[1] = 0.; rho[2] = 1.;

        // theta adjoint MUST be set
        theta_adj[0] = 0.; theta_adj[1] = 0.; theta_adj[2] = 0.;
    }
};

TEST_F(nuts_build_tree_fixture, build_tree_base_plus_no_opt_output)
{
    using namespace alg;

    auto ad_expr = ad::constant(-0.5) * 
                   (ad_vars[0] * ad_vars[0] + 
                    ad_vars[1] * ad_vars[1] +
                    ad_vars[2] * ad_vars[2]
                   ) ;
    ham = 0.; // manually computed current hamiltonian

    auto input = TreeInput(
        ad_expr, theta, theta_adj, rho, log_u, v,
        epsilon, ham
    );

    build_tree<3>(input, output, 0);

    // output optional theta/rho still unset
    EXPECT_FALSE(output.opt_theta_ref.has_value());
    EXPECT_FALSE(output.opt_rho_ref.has_value());

    // input theta properly updated
    EXPECT_DOUBLE_EQ(theta[0], -2.);
    EXPECT_DOUBLE_EQ(theta[1], 0.);
    EXPECT_DOUBLE_EQ(theta[2], 2.);
    EXPECT_DOUBLE_EQ(theta_adj[0], 2.);
    EXPECT_DOUBLE_EQ(theta_adj[1], 0.);
    EXPECT_DOUBLE_EQ(theta_adj[2], -2.);
    EXPECT_DOUBLE_EQ(rho[0], 1.);
    EXPECT_DOUBLE_EQ(rho[1], 0.);
    EXPECT_DOUBLE_EQ(rho[2], -1.);

    // check potential
    double expected_potential = -4.;
    EXPECT_EQ(output.potential, expected_potential);
    
    // theta and theta_prime should be the same
    EXPECT_EQ(theta[0], theta_prime[0]);
    EXPECT_EQ(theta[1], theta_prime[1]);
    EXPECT_EQ(theta[2], theta_prime[2]);

    EXPECT_EQ(output.n, static_cast<size_t>(0));
    EXPECT_EQ(output.s, true);

    double expected_ham = expected_potential - 0.5 * 2;
    EXPECT_EQ(output.alpha, std::min(std::exp(expected_ham - ham), 1.));
    EXPECT_EQ(output.n_alpha, static_cast<size_t>(1));
}

TEST_F(nuts_build_tree_fixture, build_tree_base_plus_opt_output)
{
    using namespace alg;

    auto ad_expr = ad::constant(-0.5) * 
                   (ad_vars[0] * ad_vars[0] + 
                    ad_vars[1] * ad_vars[1] +
                    ad_vars[2] * ad_vars[2]
                   ) ;
    ham = 0.; // manually computed current hamiltonian

    auto input = TreeInput(
        ad_expr, theta, theta_adj, rho, log_u, v,
        epsilon, ham
    );

    output.opt_theta_ref = opt_theta;
    output.opt_rho_ref = opt_rho;

    build_tree<3>(input, output, 0);

    // optional theta and rho are the same as input ones
    EXPECT_DOUBLE_EQ(opt_theta[0], theta[0]);
    EXPECT_DOUBLE_EQ(opt_theta[1], theta[1]);
    EXPECT_DOUBLE_EQ(opt_theta[2], theta[2]);
    EXPECT_DOUBLE_EQ(opt_rho[0], rho[0]);
    EXPECT_DOUBLE_EQ(opt_rho[1], rho[1]);
    EXPECT_DOUBLE_EQ(opt_rho[2], rho[2]);
}

TEST_F(nuts_build_tree_fixture, build_tree_base_plus_no_opt_output_2)
{
    using namespace alg;

    auto ad_expr = ad::constant(-0.5) * 
                   (ad_vars[0] * ad_vars[0] + 
                    ad_vars[1] * ad_vars[1] +
                    ad_vars[2] * ad_vars[2]
                   ) ;

    // different initialization (from the output of first test)
    theta[0] = -2.; theta[1] = 0.; theta[2] = 2.;
    rho[0] = 1.; rho[1] = 0.; rho[2] = -1.;
    theta_adj[0] = 2.; theta_adj[1] = 0.; theta_adj[2] = -2.;

    ham = -5.; // manually computed current hamiltonian

    auto input = TreeInput(
        ad_expr, theta, theta_adj, rho, log_u, v,
        epsilon, ham
    );

    build_tree<3>(input, output, 0);

    // input theta properly updated
    EXPECT_DOUBLE_EQ(theta[0], 4.);
    EXPECT_DOUBLE_EQ(theta[1], 0.);
    EXPECT_DOUBLE_EQ(theta[2], -4.);
    EXPECT_DOUBLE_EQ(theta_adj[0], -4.);
    EXPECT_DOUBLE_EQ(theta_adj[1], 0.);
    EXPECT_DOUBLE_EQ(theta_adj[2], 4.);
    EXPECT_DOUBLE_EQ(rho[0], -1.);
    EXPECT_DOUBLE_EQ(rho[1], 0.);
    EXPECT_DOUBLE_EQ(rho[2], 1.);

    // check potential
    double expected_potential = -16.;
    EXPECT_EQ(output.potential, expected_potential);
    
    // theta and theta_prime should be the same
    EXPECT_EQ(theta[0], theta_prime[0]);
    EXPECT_EQ(theta[1], theta_prime[1]);
    EXPECT_EQ(theta[2], theta_prime[2]);

    EXPECT_EQ(output.n, static_cast<size_t>(0));
    EXPECT_EQ(output.s, true);

    double expected_ham = expected_potential - 0.5 * 2;
    EXPECT_EQ(output.alpha, std::min(std::exp(expected_ham - ham), 1.));
    EXPECT_EQ(output.n_alpha, static_cast<size_t>(1));
}

TEST_F(nuts_build_tree_fixture, build_tree_recursion_plus_no_opt_output)
{
    using namespace alg;

    auto ad_expr = ad::constant(-0.5) * 
                   (ad_vars[0] * ad_vars[0] + 
                    ad_vars[1] * ad_vars[1] +
                    ad_vars[2] * ad_vars[2]
                   ) ;

    ham = 0.; // manually computed current hamiltonian

    auto input = TreeInput(
        ad_expr, theta, theta_adj, rho, log_u, v,
        epsilon, ham
    );

    // custom uniform distribution will always accept candidate
    // except when optimized for n'' == 0 in the recursion
    build_tree<3>(input, output, 1, [](const auto&) {return 0;});

    // input theta properly updated
    EXPECT_DOUBLE_EQ(theta[0], 4.);
    EXPECT_DOUBLE_EQ(theta[1], 0.);
    EXPECT_DOUBLE_EQ(theta[2], -4.);
    EXPECT_DOUBLE_EQ(theta_adj[0], -4.);
    EXPECT_DOUBLE_EQ(theta_adj[1], 0.);
    EXPECT_DOUBLE_EQ(theta_adj[2], 4.);
    EXPECT_DOUBLE_EQ(rho[0], -1.);
    EXPECT_DOUBLE_EQ(rho[1], 0.);
    EXPECT_DOUBLE_EQ(rho[2], 1.);

    // check potential (should not have been accepted)
    double expected_potential = -4;
    EXPECT_EQ(output.potential, expected_potential);
    
    // theta_prime should not have been accepted
    EXPECT_DOUBLE_EQ(theta_prime[0], -2.);
    EXPECT_DOUBLE_EQ(theta_prime[1], 0.);
    EXPECT_DOUBLE_EQ(theta_prime[2], 2.);

    EXPECT_EQ(output.n, static_cast<size_t>(0));
    EXPECT_EQ(output.s, false);

    double expected_ham = expected_potential - 0.5 * 2;
    // alpha = alpha' + alpha''
    EXPECT_EQ(output.alpha, std::min(std::exp(expected_ham - ham), 1.) +
                            std::min(std::exp(-16 - 0.5 * 2 - ham), 1.));
    EXPECT_EQ(output.n_alpha, static_cast<size_t>(2));
}

TEST_F(nuts_build_tree_fixture, find_reasonable_log_epsilon)
{
    auto ad_expr = ad::constant(-0.5) * 
                   (ad_vars[0] * ad_vars[0] + 
                    ad_vars[1] * ad_vars[1] +
                    ad_vars[2] * ad_vars[2]
                   ) ;
    double eps = alg::find_reasonable_epsilon<3>(ad_expr, theta, theta_adj);
    static_cast<void>(eps);
}

struct nuts_fixture : nuts_tools_fixture
{
protected:
    size_t n_adapt = 1000;
    size_t n_samples = 10000;
    size_t warmup = 10000;
    double delta = 0.6;
    size_t max_depth = 10;
    size_t seed = 4821;

    std::vector<double> w_storage, b_storage;
    Param<double> w, b;
    ppl::Data<double> x {2.5, 3, 3.5, 4, 4.5, 5.};
    ppl::Data<double> y {3.5, 4, 4.5, 5, 5.5, 6.};
    ppl::Data<double> q{2.4, 3.1, 3.6, 4, 4.5, 5.};
    ppl::Data<double> r{3.5, 4, 4.4, 5.01, 5.46, 6.1};

    nuts_fixture()
        : w_storage(n_samples)
        , b_storage(n_samples)
        , w{w_storage.data()}
        , b{b_storage.data()}
    {}

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

    nuts(model, warmup, n_samples, n_adapt, seed,
         max_depth, delta);

    plot_hist(w_storage);
    EXPECT_NEAR(sample_average(w_storage), 0., 0.1);
}

TEST_F(nuts_fixture, nuts_uniform)
{
    n_samples = 1000;
    warmup = 20000;
    n_adapt = n_samples;
    reconfigure(n_samples);
    auto model = (
        w |= uniform(0., 1.)
    );

    nuts(model, warmup, n_samples, n_adapt, seed,
         max_depth, delta);

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
    nuts(model, warmup, n_samples, n_adapt, seed,
         max_depth, delta);
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
    nuts(model, warmup, n_samples, n_adapt, seed,
         max_depth, delta);
    plot_hist(w_storage, 0.2); 
    // should be either gradually increasing then suddenly dropping to 0 or
    // suddenly jumping to 0 then gradually decreasing or
    // both of those (rare)
}

TEST_F(nuts_fixture, nuts_sample_regression_dist_weight) 
{
    reconfigure(n_samples);

    auto model = (w |= normal(0., 2.),
                  y |= normal(x * w + 1., 0.5)
    );

    nuts(model, warmup, n_samples, n_adapt, seed,
         max_depth, delta);

    plot_hist(w_storage, 0.1);
    EXPECT_NEAR(sample_average(w_storage), 1.0, 0.1);
}

TEST_F(nuts_fixture, nuts_sample_regression_dist_weight_bias) 
{
    n_adapt = 1000;
    n_samples = 1000;
    warmup = 1000;

    reconfigure(n_samples);

    auto model = (b |= normal(0., 2.),
                  w |= normal(0., 2.),
                  y |= normal(x * w + b, 0.5)
    );

    nuts(model, warmup, n_samples, n_adapt, seed,
         max_depth, delta);

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

    nuts(model, warmup, n_samples, n_adapt, seed,
         max_depth, delta);

    plot_hist(w_storage, 0.2, 0., 2.);
    plot_hist(b_storage, 0.2, 0., 2.);

    EXPECT_NEAR(sample_average(w_storage), 1.0, 0.1);
    EXPECT_NEAR(sample_average(b_storage), 1.0, 0.1);
}

TEST_F(nuts_fixture, nuts_sample_regression_fuzzy_uniform) {
    auto model = (w |= uniform(0., 2.),
                  b |= uniform(0., 2.),
                  r |= normal(q * w + b, 0.5));

    nuts(model, warmup, n_samples, n_adapt, seed,
         max_depth, delta);

    plot_hist(w_storage, 0.2, 0., 1.);
    plot_hist(b_storage, 0.2, 0., 1.);

    EXPECT_NEAR(sample_average(w_storage), 1.0, 0.1);
    EXPECT_NEAR(sample_average(b_storage), 0.95, 0.1);
}

} // namespace ppl
