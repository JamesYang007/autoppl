#include "gtest/gtest.h"
#include <cmath>
#include <array>
#include <autoppl/expression/distribution/uniform.hpp>
#include <testutil/mock_types.hpp>
#include <testutil/sample_tools.hpp>

namespace ppl {
namespace expr {

struct uniform_fixture : ::testing::Test {
protected:
    using value_t = typename MockVarExpr::value_t;
    static constexpr size_t sample_size = 1000;
    double min = -2.3;
    double max = 2.7;
    MockVarExpr x{min};
    MockVarExpr y{max};
    using unif_t = Uniform<MockVarExpr, MockVarExpr>;
    unif_t unif = {x, y};
    std::array<double, sample_size> sample = {0.};
};

TEST_F(uniform_fixture, ctor)
{
    static_assert(util::assert_is_dist_expr_v<unif_t>);
}

TEST_F(uniform_fixture, uniform_check_params) {
    EXPECT_DOUBLE_EQ(unif.min(), static_cast<value_t>(x));
    EXPECT_DOUBLE_EQ(unif.max(), static_cast<value_t>(y));
}

TEST_F(uniform_fixture, uniform_pdf_in_range)
{
    EXPECT_DOUBLE_EQ(unif.pdf(-2.2999999999), 0.2);
    EXPECT_DOUBLE_EQ(unif.pdf(-2.), 0.2);
    EXPECT_DOUBLE_EQ(unif.pdf(-1.423), 0.2);
    EXPECT_DOUBLE_EQ(unif.pdf(0.), 0.2);
    EXPECT_DOUBLE_EQ(unif.pdf(1.31), 0.2);
    EXPECT_DOUBLE_EQ(unif.pdf(2.41), 0.2);
    EXPECT_DOUBLE_EQ(unif.pdf(2.69999999999), 0.2);
}

TEST_F(uniform_fixture, uniform_pdf_out_of_range)
{
    EXPECT_DOUBLE_EQ(unif.pdf(-100), 0.);
    EXPECT_DOUBLE_EQ(unif.pdf(-3.41), 0.);
    EXPECT_DOUBLE_EQ(unif.pdf(-2.3), 0.);
    EXPECT_DOUBLE_EQ(unif.pdf(2.7), 0.);
    EXPECT_DOUBLE_EQ(unif.pdf(3.5), 0.);
    EXPECT_DOUBLE_EQ(unif.pdf(3214), 0.);
}

TEST_F(uniform_fixture, uniform_log_pdf_in_range)
{
    EXPECT_DOUBLE_EQ(unif.log_pdf(-2.2999999999), std::log(0.2));
    EXPECT_DOUBLE_EQ(unif.log_pdf(-2.), std::log(0.2));
    EXPECT_DOUBLE_EQ(unif.log_pdf(-1.423), std::log(0.2));
    EXPECT_DOUBLE_EQ(unif.log_pdf(0.), std::log(0.2));
    EXPECT_DOUBLE_EQ(unif.log_pdf(1.31), std::log(0.2));
    EXPECT_DOUBLE_EQ(unif.log_pdf(2.41), std::log(0.2));
    EXPECT_DOUBLE_EQ(unif.log_pdf(2.69999999999), std::log(0.2));
}

TEST_F(uniform_fixture, uniform_log_pdf_out_of_range)
{
    EXPECT_DOUBLE_EQ(unif.log_pdf(-100), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(unif.log_pdf(-3.41), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(unif.log_pdf(-2.3), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(unif.log_pdf(2.7), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(unif.log_pdf(3.5), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(unif.log_pdf(3214), std::numeric_limits<double>::lowest());
}

TEST_F(uniform_fixture, uniform_sample) {
    std::random_device rd{};
    std::mt19937 gen{rd()};

    for (size_t i = 0; i < sample_size; i++) {
        sample[i] = unif.sample(gen);
        EXPECT_GT(sample[i], min);
        EXPECT_LT(sample[i], max);
    }

    plot_hist(sample, 0.05, min, max);
}

} // namespace expr
} // namespace ppl
