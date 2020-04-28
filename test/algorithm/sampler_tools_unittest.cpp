#include "gtest/gtest.h"
#include <testutil/mock_types.hpp>
#include <autoppl/expr_builder.hpp>
#include <autoppl/algorithm/sampler_tools.hpp>

namespace ppl {
namespace alg {

struct sampler_tools_fixture : ::testing::Test
{
protected:
    using var_t = Param<double>;

    static constexpr size_t n_params = 10;
    std::array<Param<double>, n_params> thetas;
    Data<double> x;

    sampler_tools_fixture()
    {}
};

////////////////////////////////////////////////////////////
// hamiltonian TESTS
////////////////////////////////////////////////////////////

TEST_F(sampler_tools_fixture, hamiltonian_sanity)
{
    double potential = -3.41;
    arma::vec momentum(5, arma::fill::zeros);
    int i = 0;
    momentum.for_each([&](double& elt) {elt += (++i);});
    double actual = hamiltonian(potential, momentum);
    double expected = -3.41 - 0.5 * 55;
    EXPECT_DOUBLE_EQ(actual, expected);
}

////////////////////////////////////////////////////////////
// leapfrog TESTS
////////////////////////////////////////////////////////////

struct leapfrog_fixture : ::testing::Test
{
protected:
    static constexpr size_t n_params = 3;
    static constexpr size_t n_args = 3;
    std::vector<ad::Var<double>> v;

    // create matrix to store theta, adjoints, and momentum
    arma::mat mat;
    using submat_t = std::decay_t<decltype(mat.unsafe_col(0))>;
    submat_t theta;
    submat_t theta_adj;
    submat_t r;

    double epsilon = 2.;

    leapfrog_fixture() 
        : v(n_params)
        , mat(n_params, n_args)
        , theta(mat.unsafe_col(0))
        , theta_adj(mat.unsafe_col(1))
        , r(mat.unsafe_col(2))
    {
        // bind AD variables to theta and theta_adj
        ad_bind_storage(v, theta, theta_adj);

        // initialization of values
        // adjoints are initialized with known values to
        // test if leapfrog correctly resets adjoints if reuse is false.
        theta[0] = 1.; theta[1] = 2.; theta[2] = 3.;
        theta_adj[0] = 1.; theta_adj[1] = 2.; theta_adj[2] = 3.;
        r[0] = -1.; r[1] = 0.; r[2] = 1.;
    }
};

TEST_F(leapfrog_fixture, leapfrog_no_reuse_adj)
{
    auto ad_expr = (v[0] * v[1] + v[2]);
    double ham = leapfrog(ad_expr, theta, theta_adj, r, epsilon, false);

    EXPECT_DOUBLE_EQ(ham, 19.);
    EXPECT_DOUBLE_EQ(theta[0], 3.);
    EXPECT_DOUBLE_EQ(theta[1], 4.);
    EXPECT_DOUBLE_EQ(theta[2], 7.);
    EXPECT_DOUBLE_EQ(theta_adj[0], 4.);
    EXPECT_DOUBLE_EQ(theta_adj[1], 3.);
    EXPECT_DOUBLE_EQ(theta_adj[2], 1.);
    EXPECT_DOUBLE_EQ(r[0], 5.);
    EXPECT_DOUBLE_EQ(r[1], 4.);
    EXPECT_DOUBLE_EQ(r[2], 3.);
}

TEST_F(leapfrog_fixture, leapfrog_reuse_adj)
{
    auto ad_expr = (v[0] * v[1] + v[2]);
    double ham = leapfrog(ad_expr, theta, theta_adj, r, epsilon, true);

    EXPECT_DOUBLE_EQ(ham, 17.);
    EXPECT_DOUBLE_EQ(theta[0], 1.);
    EXPECT_DOUBLE_EQ(theta[1], 6.);
    EXPECT_DOUBLE_EQ(theta[2], 11.);
    EXPECT_DOUBLE_EQ(theta_adj[0], 6.);
    EXPECT_DOUBLE_EQ(theta_adj[1], 1.);
    EXPECT_DOUBLE_EQ(theta_adj[2], 1.);
    EXPECT_DOUBLE_EQ(r[0], 6.);
    EXPECT_DOUBLE_EQ(r[1], 3.);
    EXPECT_DOUBLE_EQ(r[2], 5.);
}

} // namespace alg
} // namespace ppl
