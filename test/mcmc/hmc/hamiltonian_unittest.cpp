#include "gtest/gtest.h"
#include <armadillo>
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
    arma::vec momentum(5, arma::fill::zeros);
    int i = 0;
    momentum.for_each([&](double& elt) {elt += (++i);});
    double kinetic = -0.5 * arma::dot(momentum, momentum);
    double actual = hamiltonian(potential, kinetic);
    double expected = -3.41 - 0.5 * 55;
    EXPECT_DOUBLE_EQ(actual, expected);
}

} // namespace mcmc
} // namespace ppl
