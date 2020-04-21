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
    double min = 0.1;
    double max = 0.8;
    MockVarExpr x{min};
    MockVarExpr y{max};
    using unif_t = Uniform<MockVarExpr, MockVarExpr>;
    unif_t unif = {x, y};
    std::array<double, sample_size> sample = {0.};
};

TEST_F(uniform_fixture, ctor)
{
    static_assert(util::is_dist_expr_v<unif_t>);
}

TEST_F(uniform_fixture, uniform_check_params) {
    EXPECT_DOUBLE_EQ(unif.min(), static_cast<value_t>(x));
    EXPECT_DOUBLE_EQ(unif.max(), static_cast<value_t>(y));
}

TEST_F(uniform_fixture, uniform_pdf_delegate) {
    EXPECT_DOUBLE_EQ(unif.pdf(-10.664), UniformBase::pdf(-10.664, min, max));
    EXPECT_DOUBLE_EQ(unif.pdf(-7.324), UniformBase::pdf(-7.324, min, max));
    EXPECT_DOUBLE_EQ(unif.pdf(-3.241), UniformBase::pdf(-3.241, min, max));
    EXPECT_DOUBLE_EQ(unif.pdf(-0.359288), UniformBase::pdf(-0.359288, min, max));
    EXPECT_DOUBLE_EQ(unif.pdf(0.12314), UniformBase::pdf(0.12314, min, max));
    EXPECT_DOUBLE_EQ(unif.pdf(3.145), UniformBase::pdf(3.145, min, max));
    EXPECT_DOUBLE_EQ(unif.pdf(6.000923), UniformBase::pdf(6.000923, min, max));
    EXPECT_DOUBLE_EQ(unif.pdf(16.423), UniformBase::pdf(16.423, min, max));
}

TEST_F(uniform_fixture, uniform_log_pdf_delegate) {
    EXPECT_DOUBLE_EQ(unif.log_pdf(-10.664), UniformBase::log_pdf(-10.664, min, max));
    EXPECT_DOUBLE_EQ(unif.log_pdf(-7.324), UniformBase::log_pdf(-7.324, min, max));
    EXPECT_DOUBLE_EQ(unif.log_pdf(-3.241), UniformBase::log_pdf(-3.241, min, max));
    EXPECT_DOUBLE_EQ(unif.log_pdf(-0.359288), UniformBase::log_pdf(-0.359288, min, max));
    EXPECT_DOUBLE_EQ(unif.log_pdf(0.12314), UniformBase::log_pdf(0.12314, min, max));
    EXPECT_DOUBLE_EQ(unif.log_pdf(3.145), UniformBase::log_pdf(3.145, min, max));
    EXPECT_DOUBLE_EQ(unif.log_pdf(6.000923), UniformBase::log_pdf(6.000923, min, max));
    EXPECT_DOUBLE_EQ(unif.log_pdf(16.423), UniformBase::log_pdf(16.423, min, max));
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
