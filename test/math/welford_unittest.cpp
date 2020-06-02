#include <autoppl/math/welford.hpp>
#include <gtest/gtest.h>

namespace ppl {
namespace math {

struct welford_fixture : ::testing::Test
{
protected:
};

TEST_F(welford_fixture, ctor)
{
    WelfordVar wel(1);
    EXPECT_EQ(wel.get_n_samples(), static_cast<size_t>(0));
}

TEST_F(welford_fixture, update_one)
{
    WelfordVar wel(3);
    arma::vec v(3), x(3, arma::fill::zeros);

    x[0] = 1;
    wel.update(x);

    EXPECT_EQ(wel.get_n_samples(), static_cast<size_t>(1));

    wel.get_variance(v);
    EXPECT_DOUBLE_EQ(v[0], 0.);
    EXPECT_DOUBLE_EQ(v[1], 0.);
    EXPECT_DOUBLE_EQ(v[2], 0.);
}

TEST_F(welford_fixture, update_two)
{
    WelfordVar wel(2);
    arma::vec v(2), x(2, arma::fill::zeros);

    x[0] = 1; x[1] = 0;
    wel.update(x);
    x[0] = 0; x[1] = 1;
    wel.update(x);

    EXPECT_EQ(wel.get_n_samples(), static_cast<size_t>(2));

    wel.get_variance(v);
    EXPECT_DOUBLE_EQ(v[0], 0.25);
    EXPECT_DOUBLE_EQ(v[1], 0.25);
}

} // namespace math
} // namespace ppl
