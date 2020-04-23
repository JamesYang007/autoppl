#include "gtest/gtest.h"
#include <cmath>
#include <array>
#include <autoppl/expression/distribution/normal.hpp>
#include <testutil/mock_types.hpp>
#include <testutil/sample_tools.hpp>

namespace ppl {
namespace expr {

struct normal_fixture : ::testing::Test {
protected:
    using value_t = typename MockVarExpr::value_t;
    static constexpr size_t sample_size = 1000;
    double mean = 0.3;
    double stddev = 1.3;
    double tol = 1e-15;
    MockVarExpr x{mean};
    MockVarExpr y{stddev};
    using norm_t = Normal<MockVarExpr, MockVarExpr>;
    norm_t norm = {x, y};
    std::array<double, sample_size> sample = {0.};
};

TEST_F(normal_fixture, ctor)
{
    static_assert(util::assert_is_dist_expr_v<norm_t>);
}

TEST_F(normal_fixture, normal_check_params) {
    EXPECT_DOUBLE_EQ(norm.mean(), static_cast<value_t>(x));
    EXPECT_DOUBLE_EQ(norm.stddev(), static_cast<value_t>(y));
}

TEST_F(normal_fixture, normal_pdf)
{
    EXPECT_NEAR(norm.pdf(-10.231), 1.726752595588348216742E-15, tol);
    EXPECT_NEAR(norm.pdf(-5.31), 2.774166877919518907166E-5, tol);
    EXPECT_DOUBLE_EQ(norm.pdf(-2.3141231), 0.04063645713784323551341);
    EXPECT_DOUBLE_EQ(norm.pdf(0.), 0.2988151821496727914542);
    EXPECT_DOUBLE_EQ(norm.pdf(1.31), 0.2269313951019926611687);
    EXPECT_DOUBLE_EQ(norm.pdf(3.21), 0.02505560241243631472997);
    EXPECT_NEAR(norm.pdf(5.24551), 2.20984513448306056291E-4, tol);
    EXPECT_NEAR(norm.pdf(10.5699), 8.61135160183067521907E-15, tol);
}

TEST_F(normal_fixture, normal_log_pdf)
{
    EXPECT_DOUBLE_EQ(norm.log_pdf(-10.231), std::log(1.726752595588348216742E-15));
    EXPECT_DOUBLE_EQ(norm.log_pdf(-5.31), std::log(2.774166877919518907166E-5));
    EXPECT_DOUBLE_EQ(norm.log_pdf(-2.3141231), std::log(0.04063645713784323551341));
    EXPECT_DOUBLE_EQ(norm.log_pdf(0.), std::log(0.2988151821496727914542));
    EXPECT_DOUBLE_EQ(norm.log_pdf(1.31), std::log(0.2269313951019926611687));
    EXPECT_DOUBLE_EQ(norm.log_pdf(3.21), std::log(0.02505560241243631472997));
    EXPECT_DOUBLE_EQ(norm.log_pdf(5.24551), std::log(2.20984513448306056291E-4));
    EXPECT_DOUBLE_EQ(norm.log_pdf(10.5699), std::log(8.61135160183067521907E-15));
}

TEST_F(normal_fixture, normal_sample) {
    std::random_device rd{};
    std::mt19937 gen{rd()};

    for (size_t i = 0; i < sample_size; i++) {
        sample[i] = norm.sample(gen);
    }

    plot_hist(sample);
}

} // namespace expr
} // namespace ppl
