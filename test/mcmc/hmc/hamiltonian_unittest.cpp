#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <autoppl/mcmc/hmc/hamiltonian.hpp>

namespace ppl {
namespace mcmc {

struct hamiltonian_fixture : ::testing::Test
{
protected:
};

////////////////////////////////////////////////////////////
// hamiltonian TESTS
////////////////////////////////////////////////////////////

TEST_F(hamiltonian_fixture, hamiltonian_sanity)
{
    double potential = -3.41;
    Eigen::VectorXd momentum(5);
    momentum.setZero();
    momentum += Eigen::VectorXd::LinSpaced(momentum.size(), 1, momentum.size());
    double kinetic = -0.5 * momentum.squaredNorm();
    double actual = hamiltonian(potential, kinetic);
    double expected = -3.41 - 0.5 * 55;
    EXPECT_DOUBLE_EQ(actual, expected);
}

} // namespace mcmc
} // namespace ppl
