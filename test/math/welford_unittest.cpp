#include <autoppl/math/welford.hpp>
#include <gtest/gtest.h>

namespace ppl {
namespace math {

struct welford_fixture : ::testing::Test
{
protected:
    using vec_t = Eigen::VectorXd;
};

TEST_F(welford_fixture, ctor)
{
    WelfordVar wel(1);
    EXPECT_EQ(wel.get_n_samples(), static_cast<size_t>(0));
}

TEST_F(welford_fixture, update_one)
{
    WelfordVar wel(3);
    vec_t v(3), x(3);
    x.setZero();

    x[0] = 1;
    wel.update(x);

    EXPECT_EQ(wel.get_n_samples(), static_cast<size_t>(1));

    v = wel.get_variance();
    EXPECT_DOUBLE_EQ(v[0], 0.);
    EXPECT_DOUBLE_EQ(v[1], 0.);
    EXPECT_DOUBLE_EQ(v[2], 0.);
}

TEST_F(welford_fixture, update_two)
{
    WelfordVar wel(2);
    vec_t v(2), x(2);

    x[0] = 1; x[1] = 0;
    wel.update(x);
    x[0] = 0; x[1] = 1;
    wel.update(x);

    EXPECT_EQ(wel.get_n_samples(), static_cast<size_t>(2));

    v = wel.get_variance() / (wel.get_n_samples() - 1);
    EXPECT_DOUBLE_EQ(v[0], 0.25);
    EXPECT_DOUBLE_EQ(v[1], 0.25);
}

} // namespace math
} // namespace ppl
