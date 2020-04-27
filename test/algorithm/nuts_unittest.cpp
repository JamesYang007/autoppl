#include "gtest/gtest.h" 
#include <array>
#include <autoppl/expr_builder.hpp>
#include <autoppl/algorithm/nuts.hpp>
#include <testutil/sample_tools.hpp>
#include <fastad>

namespace ppl {

struct nuts_fixture : ::testing::Test
{
protected:
    static constexpr size_t n = 10; // number of variables
    using ad_vec_t = std::vector<ad::Var<double>>;
    std::array<Variable<double>, n> thetas; // model variables
    std::vector<const void*> keys; // keys for building ad expr
    ad_vec_t theta_minus; // ad variables
    ad_vec_t theta_plus; // ad variables

    arma::mat rho_mat; // momentum matrix
    arma::mat theta_mat; // other auxiliary theta matrix
    using subview_t = std::decay_t<decltype(rho_mat.col(0))>;
    subview_t rho_minus;
    subview_t rho_plus;
    subview_t theta_prime;

    double log_u = 0.;
    int8_t v = -1;
    size_t j = 0;
    double eps_prev = 2.;
    double ham_prev = -4.32;

    using output_t = alg::TreeOutput<ad_vec_t, subview_t>;
    output_t output;

    nuts_fixture()
        : thetas({0., 1.})
        , keys({&thetas[0], &thetas[1]})
        , theta_minus(n)
        , theta_plus(n)
        , rho_mat(n,2)
        , theta_mat(n,1)
        , rho_minus{rho_mat.col(0)}
        , rho_plus{rho_mat.col(1)}
        , theta_prime{theta_mat.col(0)}
    {
        theta_minus[0].set_value(0.);
        theta_minus[1].set_value(1.);
        theta_plus[0].set_value(0.);
        theta_plus[1].set_value(1.);
        theta_prime[0] = -131.;
        theta_prime[1] = 5.24113;
        rho_minus[0] = 1.;
        rho_minus[1] = 2.;
        rho_plus[0] = 1.;
        rho_plus[1] = 2.;
    }

    template <class ArrayType>
    double sample_average(const ArrayType& storage, size_t burn)
    {
        double sum = std::accumulate(
                std::next(storage.begin(), burn),
                storage.end(), 
                0.);
        return sum / (storage.size() - burn);
    }
};


TEST_F(nuts_fixture, ad_copy_values_varvec_mat)
{
    using namespace alg;
    
    std::vector<ad::Var<double>> dest(2);
    arma::mat src(2,2);
    src(0,0) = 3.1; src(0,1) = 3.2;
    src(1,0) = 1.0; src(1,1) = 2.1;

    ad_copy_values(dest, src.col(0));
    EXPECT_EQ(dest[0].get_value(), src(0,0));
    EXPECT_EQ(dest[1].get_value(), src(1,0));
    EXPECT_EQ(dest[0].get_adjoint(), 0.0);
    EXPECT_EQ(dest[1].get_adjoint(), 0.0);

    ad_copy_values(dest, src.col(1));
    EXPECT_EQ(dest[0].get_value(), src(0,1));
    EXPECT_EQ(dest[1].get_value(), src(1,1));
    EXPECT_EQ(dest[0].get_adjoint(), 0.0);
    EXPECT_EQ(dest[1].get_adjoint(), 0.0);
}

TEST_F(nuts_fixture, ad_copy_values_varvec_mat_fail)
{
    using namespace alg;
    std::vector<ad::Var<double>> dest(2);
    arma::mat src(3,2);
    auto col = src.col(0);
    ASSERT_DEATH(ad_copy_values(dest, col), "");
}

TEST_F(nuts_fixture, ad_copy_values_mat_varvec)
{
    using namespace alg;
    
    std::vector<ad::Var<double>> src(2);
    src[0].set_value(-0.3);
    src[1].set_value(1.2);
    arma::mat dest(2,3);
    auto rho1 = dest.col(0);
    auto rho2 = dest.col(1);

    ad_copy_values(rho1, src);
    EXPECT_EQ(rho1[0], -0.3);
    EXPECT_EQ(rho1[1], 1.2);

    src[0].set_value(3.58);
    src[1].set_value(-0.32);

    ad_copy_values(rho2, src);
    EXPECT_EQ(rho2[0], 3.58);
    EXPECT_EQ(rho2[1], -0.32);
}

TEST_F(nuts_fixture, ad_copy_values_mat_varvec_fail)
{
    using namespace alg;
    std::vector<ad::Var<double>> src(2);
    arma::mat dest(3,2);
    auto rho1 = dest.col(0);
    ASSERT_DEATH(ad_copy_values(rho1, src), "");
}

TEST_F(nuts_fixture, ad_copy_values_varvec_varvec)
{
    using namespace alg;
    
    std::vector<ad::Var<double>> dest(2);
    std::vector<ad::Var<double>> src(2);
    src[0].set_value(-0.3);
    src[1].set_value(1.2);

    ad_copy_values(dest, src);
    EXPECT_EQ(dest[0].get_value(), -0.3);
    EXPECT_EQ(dest[1].get_value(), 1.2);
    EXPECT_EQ(dest[0].get_adjoint(), 0.0);
    EXPECT_EQ(dest[1].get_adjoint(), 0.0);

    src[0].set_value(-0.8);
    src[1].set_value(5.2);
    ad_copy_values(dest, src);
    EXPECT_EQ(dest[0].get_value(), -0.8);
    EXPECT_EQ(dest[1].get_value(), 5.2);
    EXPECT_EQ(dest[0].get_adjoint(), 0.0);
    EXPECT_EQ(dest[1].get_adjoint(), 0.0);
}

TEST_F(nuts_fixture, ad_copy_values_varvec_varvec_fail)
{
    using namespace alg;
    std::vector<ad::Var<double>> dest(2);
    std::vector<ad::Var<double>> src(3);
    ASSERT_DEATH(ad_copy_values(dest, src), "");
}

TEST_F(nuts_fixture, check_slice_1d)
{
    using namespace alg;
    std::vector<ad::Var<double>> theta_plus(1); 
    theta_plus[0].set_value(3.);
    std::vector<ad::Var<double>> theta_minus(1);
    theta_minus[0].set_value(-2.3);
    arma::mat rho(1, 2);
    rho(0,0) = -0.4; rho(0,1) = 3.2;
    auto rho_plus = rho.col(0);
    auto rho_minus = rho.col(1);

    bool actual = check_slice(theta_plus, theta_minus, rho_plus, rho_minus);
    EXPECT_FALSE(actual);

    rho(0,0) *= -1;
    actual = check_slice(theta_plus, theta_minus, rho_plus, rho_minus);
    EXPECT_TRUE(actual);
}

TEST_F(nuts_fixture, check_slice_3d)
{
    using namespace alg;
    std::vector<ad::Var<double>> theta_plus(3); 
    theta_plus[0].set_value(3.);
    theta_plus[1].set_value(1.);
    theta_plus[2].set_value(2.);

    std::vector<ad::Var<double>> theta_minus(3);
    theta_minus[0].set_value(-2.3);
    theta_minus[0].set_value(4.2);
    theta_minus[0].set_value(-1.7);

    arma::mat rho(3, 2);
    rho(0,0) = -1.4; rho(0,1) = 0.2;
    rho(1,0) = -5.2; rho(1,1) = 1.3;
    rho(2,0) = 10.2; rho(2,1) = 2.2;
    auto rho_plus = rho.col(0);
    auto rho_minus = rho.col(1);

    bool actual = check_slice(theta_plus, theta_minus, rho_plus, rho_minus);
    EXPECT_TRUE(actual);

    rho(1,0) *= -1;
    rho(2,0) *= -1;
    actual = check_slice(theta_plus, theta_minus, rho_plus, rho_minus);
    EXPECT_FALSE(actual);
}

TEST_F(nuts_fixture, swap) 
{
    using namespace alg;

    std::vector<ad::Var<double>> v(3);
    v[0].set_value(0.3);
    v[1].set_value(-1.3);
    v[2].set_value(5.3);
    std::vector<ad::Var<double>> v2(3);
    v2[0].set_value(1.);
    v2[1].set_value(2.);
    v2[2].set_value(3.);
    swap(v, v2);

    EXPECT_EQ(v[0].get_value(), 1.);
    EXPECT_EQ(v[1].get_value(), 2.);
    EXPECT_EQ(v[2].get_value(), 3.);

    EXPECT_EQ(v2[0].get_value(), 0.3);
    EXPECT_EQ(v2[1].get_value(), -1.3);
    EXPECT_EQ(v2[2].get_value(), 5.3);
}

TEST_F(nuts_fixture, leapfrog)
{
    using namespace alg;

    auto model = (
        thetas[0] |= normal(0., 2.),
        thetas[1] |= normal(thetas[0], 1.)
    );
    std::vector<ad::Var<double>> theta_ad(2);
    std::vector<ad::Var<double>> out_theta(2);
    arma::mat rho_mat(2, 2);
    auto rho = rho_mat.col(0);
    auto out_rho = rho_mat.col(1);
    double step = 2.;

    theta_ad[0].set_value(0.);
    theta_ad[1].set_value(1.);

    rho[0] = 1.; rho[1] = 2.;

    auto theta_ad_expr = model.ad_log_pdf(keys, theta_ad);
    bool theta_ad_adjoint_exists = false;

    auto [ham_new, potential_new] = leapfrog(
            std::ref(theta_ad_expr), std::ref(theta_ad),
            theta_ad_adjoint_exists, std::ref(rho), step, 
            std::ref(out_theta), std::ref(out_rho));

    // theta_ad values should not have changed
    EXPECT_EQ(theta_ad[0].get_value(), 0.);
    EXPECT_EQ(theta_ad[1].get_value(), 1.);

    // out_theta and out_rho properly leaped
    EXPECT_EQ(out_theta[0].get_value(), 4.);
    EXPECT_EQ(out_theta[1].get_value(), 3.);
    EXPECT_EQ(out_rho[0], 0.);
    EXPECT_EQ(out_rho[1], 2.);

    // potential energy and hamiltonian
    EXPECT_EQ(potential_new, -2.5 - std::log(2));
    EXPECT_EQ(ham_new, -2.5 - std::log(2) - 2.);
}

TEST_F(nuts_fixture, build_tree_base_minus_only)
{
    using namespace alg;
    auto model = (
        thetas[0] |= normal(0., 2.),
        thetas[1] |= normal(thetas[0], 1.)
    );
    auto theta_minus_ad = model.ad_log_pdf(keys, theta_minus);
    auto input = TreeInput(
        theta_minus_ad, theta_minus, rho_minus, log_u, v, j,
        eps_prev, ham_prev
    );
    output.theta_minus_ref = theta_minus;
    output.rho_minus_ref = rho_minus;
    output.theta_prime_ref = theta_prime;

    build_tree(model, input, output);

    // theta_minus, theta_prime are set and the same
    // all pluses should still be optional (unset)
    EXPECT_TRUE(output.theta_minus_ref.has_value());
    EXPECT_TRUE(output.rho_minus_ref.has_value());
    EXPECT_TRUE(output.theta_prime_ref.has_value());
    EXPECT_FALSE(output.theta_plus_ref.has_value());
    EXPECT_FALSE(output.rho_plus_ref.has_value());

    EXPECT_EQ(theta_prime[0], 0.);
    EXPECT_EQ(theta_prime[1], -5.);
    EXPECT_EQ(rho_minus[0], 5.);
    EXPECT_EQ(rho_minus[1], -2.);
    EXPECT_EQ(theta_minus[0].get_value(), theta_prime[0]);
    EXPECT_EQ(theta_minus[1].get_value(), theta_prime[1]);

    EXPECT_EQ(output.n_prime, static_cast<size_t>(0));
    EXPECT_EQ(output.s_prime, true);

    double expected_potential = -(12.5 + std::log(2));
    EXPECT_EQ(output.potential_curr, expected_potential);
    EXPECT_EQ(output.alpha, std::exp( (expected_potential - 0.5 * 29) - ham_prev));
    EXPECT_EQ(output.n_alpha, static_cast<size_t>(1));
}

TEST_F(nuts_fixture, build_tree_base_minus_plus)
{
    using namespace alg;
    auto model = (
        thetas[0] |= normal(0., 2.),
        thetas[1] |= normal(thetas[0], 1.)
    );
    auto theta_minus_ad = model.ad_log_pdf(keys, theta_minus);
    auto input = TreeInput(
        theta_minus_ad, theta_minus, rho_minus, log_u, v, j,
        eps_prev, ham_prev
    );
    output.theta_minus_ref = theta_minus;
    output.rho_minus_ref = rho_minus;
    output.theta_plus_ref = theta_plus;
    output.rho_plus_ref = rho_plus;
    output.theta_prime_ref = theta_prime;

    build_tree(model, input, output);

    // everything is still set
    EXPECT_TRUE(output.theta_minus_ref.has_value());
    EXPECT_TRUE(output.rho_minus_ref.has_value());
    EXPECT_TRUE(output.theta_prime_ref.has_value());
    EXPECT_TRUE(output.theta_plus_ref.has_value());
    EXPECT_TRUE(output.rho_plus_ref.has_value());

    EXPECT_EQ(theta_prime[0], 0.);
    EXPECT_EQ(theta_prime[1], -5.);
    EXPECT_EQ(rho_minus[0], 5.);
    EXPECT_EQ(rho_minus[1], -2.);
    EXPECT_EQ(theta_minus[0].get_value(), theta_prime[0]);
    EXPECT_EQ(theta_minus[1].get_value(), theta_prime[1]);
    EXPECT_EQ(theta_plus[0].get_value(), theta_prime[0]);
    EXPECT_EQ(theta_plus[1].get_value(), theta_prime[1]);

    EXPECT_EQ(output.n_prime, static_cast<size_t>(0));
    EXPECT_EQ(output.s_prime, true);

    double expected_potential = -(12.5 + std::log(2));
    EXPECT_EQ(output.potential_curr, expected_potential);
    EXPECT_EQ(output.alpha, std::exp( (expected_potential - 0.5 * 29) - ham_prev));
    EXPECT_EQ(output.n_alpha, static_cast<size_t>(1));
}

TEST_F(nuts_fixture, build_tree_base_minus_only_adj_exists)
{
    using namespace alg;
    auto model = (
        thetas[0] |= normal(0., 2.),
        thetas[1] |= normal(thetas[0], 1.)
    );
    auto theta_minus_ad = model.ad_log_pdf(keys, theta_minus);
    auto input = TreeInput(
        theta_minus_ad, theta_minus, rho_minus, log_u, v, j,
        eps_prev, ham_prev
    );

    theta_minus[0].set_adjoint(1.);
    theta_minus[1].set_adjoint(-1.);
    input.theta_adjoint_exists = true;

    output.theta_minus_ref = theta_minus;
    output.rho_minus_ref = rho_minus;
    output.theta_prime_ref = theta_prime;

    build_tree(model, input, output);

    // theta_minus, theta_prime are set and the same
    // all pluses should still be optional (unset)
    EXPECT_TRUE(output.theta_minus_ref.has_value());
    EXPECT_TRUE(output.rho_minus_ref.has_value());
    EXPECT_TRUE(output.theta_prime_ref.has_value());
    EXPECT_FALSE(output.theta_plus_ref.has_value());
    EXPECT_FALSE(output.rho_plus_ref.has_value());

    EXPECT_EQ(theta_prime[0], 0.);
    EXPECT_EQ(theta_prime[1], -5.);
    EXPECT_EQ(rho_minus[0], 5.);
    EXPECT_EQ(rho_minus[1], -2.);
    EXPECT_EQ(theta_minus[0].get_value(), theta_prime[0]);
    EXPECT_EQ(theta_minus[1].get_value(), theta_prime[1]);

    EXPECT_EQ(output.n_prime, static_cast<size_t>(0));
    EXPECT_EQ(output.s_prime, true);

    double expected_potential = -(12.5 + std::log(2));
    EXPECT_EQ(output.potential_curr, expected_potential);
    EXPECT_EQ(output.alpha, std::exp((expected_potential - 0.5 * 29) - ham_prev));
    EXPECT_EQ(output.n_alpha, static_cast<size_t>(1));
}

TEST_F(nuts_fixture, build_tree_base_plus_only)
{
    using namespace alg;
    v = 1;
    auto model = (
        thetas[0] |= normal(0., 2.),
        thetas[1] |= normal(thetas[0], 1.)
    );
    auto theta_plus_ad = model.ad_log_pdf(keys, theta_plus);
    auto input = TreeInput(
        theta_plus_ad, theta_plus, rho_plus, log_u, v, j,
        eps_prev, ham_prev
    );

    output.theta_plus_ref = theta_plus;
    output.rho_plus_ref = rho_plus;
    output.theta_prime_ref = theta_prime;

    build_tree(model, input, output);

    // theta_plus, theta_prime are set and the same
    // all pluses should still be optional (unset)
    EXPECT_TRUE(output.theta_plus_ref.has_value());
    EXPECT_TRUE(output.rho_plus_ref.has_value());
    EXPECT_TRUE(output.theta_prime_ref.has_value());
    EXPECT_FALSE(output.theta_minus_ref.has_value());
    EXPECT_FALSE(output.rho_minus_ref.has_value());

    EXPECT_EQ(theta_prime[0], 4.);
    EXPECT_EQ(theta_prime[1], 3.);
    EXPECT_EQ(rho_plus[0], 0.);
    EXPECT_EQ(rho_plus[1], 2.);
    EXPECT_EQ(theta_plus[0].get_value(), theta_prime[0]);
    EXPECT_EQ(theta_plus[1].get_value(), theta_prime[1]);

    EXPECT_EQ(output.n_prime, static_cast<size_t>(0));
    EXPECT_EQ(output.s_prime, true);

    double expected_potential = -(2.5 + std::log(2));
    EXPECT_EQ(output.potential_curr, expected_potential);
    EXPECT_EQ(output.alpha, std::exp((expected_potential - 2) - ham_prev));
    EXPECT_EQ(output.n_alpha, static_cast<size_t>(1));
}

TEST_F(nuts_fixture, build_tree_recursion_minus_only)
{
    using namespace alg;
    j = 1;
    auto model = (
        thetas[0] |= normal(0., 2.),
        thetas[1] |= normal(thetas[0], 1.)
    );
    auto theta_minus_ad = model.ad_log_pdf(keys, theta_minus);
    auto input = TreeInput(
        theta_minus_ad, theta_minus, rho_minus, log_u, v, j,
        eps_prev, ham_prev
    );
    output.theta_minus_ref = theta_minus;
    output.rho_minus_ref = rho_minus;
    output.theta_prime_ref = theta_prime;

    // dummy initialization to demonstrate that these are properly
    // set by calling build_tree
    output.alpha = -4214;
    output.n_alpha = 144;
    output.n_prime = 1345;
    output.s_prime = false;

    build_tree(model, input, output);

    // theta_minus, theta_prime are set
    // all pluses should still be optional (unset)
    EXPECT_TRUE(output.theta_minus_ref.has_value());
    EXPECT_TRUE(output.rho_minus_ref.has_value());
    EXPECT_TRUE(output.theta_prime_ref.has_value());
    EXPECT_FALSE(output.theta_plus_ref.has_value());
    EXPECT_FALSE(output.rho_plus_ref.has_value());

    EXPECT_EQ(theta_minus[0].get_value(), -20.);
    EXPECT_EQ(theta_minus[1].get_value(), 9.);
    EXPECT_EQ(rho_minus[0], -24.);
    EXPECT_EQ(rho_minus[1], 22.);

    EXPECT_EQ(output.n_prime, static_cast<size_t>(0));
    EXPECT_EQ(output.s_prime, false);

    double expected_ham_1 = -(12.5 + std::log(2) + 0.5 * 29);
    double alpha_prime = std::exp(expected_ham_1 - ham_prev);
    double expected_potential_2 = -0.5*(29*29 + 2*std::log(2) + 5*20);
    alpha_prime += std::exp( (expected_potential_2 -0.5*(34*34 + 29*29)) - ham_prev);
    EXPECT_EQ(output.alpha, alpha_prime);
    EXPECT_EQ(output.n_alpha, static_cast<size_t>(2));
}

TEST_F(nuts_fixture, build_tree_recursion_minus_plus)
{
    using namespace alg;
    j = 1;
    auto model = (
        thetas[0] |= normal(0., 2.),
        thetas[1] |= normal(thetas[0], 1.)
    );
    auto theta_minus_ad = model.ad_log_pdf(keys, theta_minus);
    auto input = TreeInput(
        theta_minus_ad, theta_minus, rho_minus, log_u, v, j,
        eps_prev, ham_prev
    );
    output.theta_minus_ref = theta_minus;
    output.rho_minus_ref = rho_minus;
    output.theta_plus_ref = theta_plus;
    output.rho_plus_ref = rho_plus;
    output.theta_prime_ref = theta_prime;

    // dummy initialization to demonstrate that these are properly
    // set by calling build_tree
    output.alpha = -4214;
    output.n_alpha = 144;
    output.n_prime = 1345;
    output.s_prime = false;

    build_tree(model, input, output);

    // everything still set
    EXPECT_TRUE(output.theta_minus_ref.has_value());
    EXPECT_TRUE(output.rho_minus_ref.has_value());
    EXPECT_TRUE(output.theta_prime_ref.has_value());
    EXPECT_TRUE(output.theta_plus_ref.has_value());
    EXPECT_TRUE(output.rho_plus_ref.has_value());

    EXPECT_EQ(theta_minus[0].get_value(), -20.);
    EXPECT_EQ(theta_minus[1].get_value(), 9.);
    EXPECT_EQ(theta_plus[0].get_value(), 0.);
    EXPECT_EQ(theta_plus[1].get_value(), -5.);
    EXPECT_EQ(rho_minus[0], -24.);
    EXPECT_EQ(rho_minus[1], 22.);

    EXPECT_EQ(output.n_prime, static_cast<size_t>(0));
    EXPECT_EQ(output.s_prime, false);

    double expected_ham_1 = -(12.5 + std::log(2) + 0.5 * 29);
    double alpha_prime = std::exp(expected_ham_1 - ham_prev);
    double expected_potential_2 = -0.5*(29*29 + 2*std::log(2) + 5*20);
    alpha_prime += std::exp( (expected_potential_2 -0.5*(34*34 + 29*29)) - ham_prev);
    EXPECT_EQ(output.alpha, alpha_prime);
    EXPECT_EQ(output.n_alpha, static_cast<size_t>(2));
}

TEST_F(nuts_fixture, nuts)
{
    constexpr size_t n_samples = 2000;
    constexpr size_t n_adapt = 10;
    double delta = 0.65;
    using state_t = typename Variable<double>::state_t;

    std::vector<double> samples_0(n_samples);
    thetas[0].set_state(state_t::parameter);
    thetas[0].set_storage(samples_0.data());

    std::vector<double> samples_1(n_samples);
    thetas[1].set_state(state_t::parameter);
    thetas[1].set_storage(samples_1.data());

    //Variable<double> x;
    //x.observe(3.);

    auto model = (
        thetas[0] |= normal(0., 1.)
        //thetas[1] |= normal(0., 1.)
        //x |= normal(thetas[0], 1.)
    );

    size_t max_depth = 10;
    size_t max_init_iter = 4;
    size_t seed = 4821;
    nuts(model, delta, n_samples, n_adapt, max_depth, max_init_iter, seed);

    std::vector<double> burnt_samples(
            std::next(samples_0.begin(), n_samples/2),
            samples_0.end());
    plot_hist(burnt_samples, 0.25);
    size_t burn = 1000;
    EXPECT_NEAR(sample_average(samples_0, burn), 0., 0.1);
}

} // namespace ppl
