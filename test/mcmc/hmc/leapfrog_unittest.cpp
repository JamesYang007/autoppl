#include "gtest/gtest.h"
#include <vector>
#include <fastad>
#include <autoppl/mcmc/hmc/leapfrog.hpp>
#include <autoppl/mcmc/hmc/momentum_handler.hpp>

namespace ppl {
namespace mcmc {

////////////////////////////////////////////////////////////
// leapfrog TESTS
////////////////////////////////////////////////////////////

struct leapfrog_fixture : ::testing::Test
{
protected:
    static constexpr size_t n_params = 3;
    static constexpr size_t n_args = 4;
    std::vector<ad::Var<double>> v;

    // create matrix to store theta, adjoints, and momentum
    Eigen::MatrixXd mat;
    using submat_t = std::decay_t<decltype(mat.col(0))>;
    submat_t theta;
    submat_t theta_adj;
    submat_t tp_adj;
    submat_t r;

    MomentumHandler<unit_var> m_handler;

    double epsilon = 2.;

    leapfrog_fixture() 
        : v(n_params)
        , mat(n_params, n_args)
        , theta(mat.col(0))
        , theta_adj(mat.col(1))
        , tp_adj(mat.col(2))
        , r(mat.col(3))
    {
        for (size_t i = 0; i < v.size(); ++i) {
            v[i].bind(&theta[i]);
            v[i].bind_adj(&theta_adj[i]);
        }

        // initialization of values
        // adjoints are initialized with known values to
        // test if leapfrog correctly resets adjoints if reuse is false.
        theta[0] = 1.; theta[1] = 2.; theta[2] = 3.;
        theta_adj[0] = 1.; theta_adj[1] = 2.; theta_adj[2] = 3.;
        r[0] = -1.; r[1] = 0.; r[2] = 1.;

        tp_adj.setZero();
    }
};

TEST_F(leapfrog_fixture, leapfrog_no_reuse_adj)
{
    auto ad_expr = ad::bind(v[0] * v[1] + v[2]);
    double ham = leapfrog(
            ad_expr, theta, theta_adj, tp_adj,
            r, m_handler, epsilon, false);

    EXPECT_DOUBLE_EQ(ham, -19.);
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
    auto ad_expr = ad::bind(v[0] * v[1] + v[2]);
    double ham = leapfrog(
            ad_expr, theta, theta_adj, tp_adj,
            r, m_handler, epsilon, true);

    EXPECT_DOUBLE_EQ(ham, -17.);
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

} // namespace mcmc
} // namespace ppl
