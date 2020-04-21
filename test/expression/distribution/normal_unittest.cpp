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
    double mean = 0.1;
    double stddev = 0.8;
    MockVarExpr x{mean};
    MockVarExpr y{stddev};
    using norm_t = Normal<MockVarExpr, MockVarExpr>;
    norm_t norm = {x, y};
    std::array<double, sample_size> sample = {0.};
};

TEST_F(normal_fixture, ctor)
{
    static_assert(util::is_dist_expr_v<norm_t>);
}

TEST_F(normal_fixture, normal_check_params) {
    EXPECT_DOUBLE_EQ(norm.mean(), static_cast<value_t>(x));
    EXPECT_DOUBLE_EQ(norm.stddev(), static_cast<value_t>(y));
}

TEST_F(normal_fixture, normal_pdf_delegate) {
    EXPECT_DOUBLE_EQ(norm.pdf(-10.664), NormalBase::pdf(-10.664, mean, stddev));
    EXPECT_DOUBLE_EQ(norm.pdf(-7.324), NormalBase::pdf(-7.324, mean, stddev));
    EXPECT_DOUBLE_EQ(norm.pdf(-3.241), NormalBase::pdf(-3.241, mean, stddev));
    EXPECT_DOUBLE_EQ(norm.pdf(-0.359288), NormalBase::pdf(-0.359288, mean, stddev));
    EXPECT_DOUBLE_EQ(norm.pdf(0.12314), NormalBase::pdf(0.12314, mean, stddev));
    EXPECT_DOUBLE_EQ(norm.pdf(3.145), NormalBase::pdf(3.145, mean, stddev));
    EXPECT_DOUBLE_EQ(norm.pdf(6.000923), NormalBase::pdf(6.000923, mean, stddev));
    EXPECT_DOUBLE_EQ(norm.pdf(16.423), NormalBase::pdf(16.423, mean, stddev));
}

TEST_F(normal_fixture, normal_log_pdf_delegate) {
    EXPECT_DOUBLE_EQ(norm.log_pdf(-10.664), NormalBase::log_pdf(-10.664, mean, stddev));
    EXPECT_DOUBLE_EQ(norm.log_pdf(-7.324), NormalBase::log_pdf(-7.324, mean, stddev));
    EXPECT_DOUBLE_EQ(norm.log_pdf(-3.241), NormalBase::log_pdf(-3.241, mean, stddev));
    EXPECT_DOUBLE_EQ(norm.log_pdf(-0.359288), NormalBase::log_pdf(-0.359288, mean, stddev));
    EXPECT_DOUBLE_EQ(norm.log_pdf(0.12314), NormalBase::log_pdf(0.12314, mean, stddev));
    EXPECT_DOUBLE_EQ(norm.log_pdf(3.145), NormalBase::log_pdf(3.145, mean, stddev));
    EXPECT_DOUBLE_EQ(norm.log_pdf(6.000923), NormalBase::log_pdf(6.000923, mean, stddev));
    EXPECT_DOUBLE_EQ(norm.log_pdf(16.423), NormalBase::log_pdf(16.423, mean, stddev));
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
