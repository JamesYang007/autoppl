#include "gtest/gtest.h"
#include <cmath>
#include <array>
#include <autoppl/expression/distribution/bernoulli.hpp>
#include <testutil/mock_types.hpp>
#include <testutil/sample_tools.hpp>

namespace ppl {
namespace expr {

struct bernoulli_fixture : ::testing::Test 
{
protected:
    using value_t = typename MockVarExpr::value_t;
    static constexpr size_t sample_size = 1000;
    double p = 0.6;
    MockVarExpr x{p};
    Bernoulli<MockVarExpr> bern = {x};
    std::array<double, sample_size> sample = {0.};
};

TEST_F(bernoulli_fixture, ctor)
{
    static_assert(util::assert_is_dist_expr_v<Bernoulli<MockVarExpr>>);
}

TEST_F(bernoulli_fixture, bernoulli_check_params) {
    EXPECT_DOUBLE_EQ(bern.p(), static_cast<value_t>(x));
}

TEST_F(bernoulli_fixture, bernoulli_pdf_in_range)
{
    EXPECT_DOUBLE_EQ(bern.pdf(0), 1-p);
    EXPECT_DOUBLE_EQ(bern.pdf(1), p);
}

TEST_F(bernoulli_fixture, bernoulli_pdf_out_of_range)
{
    EXPECT_DOUBLE_EQ(bern.pdf(-100), 0.);
    EXPECT_DOUBLE_EQ(bern.pdf(-3), 0.);
    EXPECT_DOUBLE_EQ(bern.pdf(-2), 0.);
    EXPECT_DOUBLE_EQ(bern.pdf(2), 0.);
    EXPECT_DOUBLE_EQ(bern.pdf(3), 0.);
    EXPECT_DOUBLE_EQ(bern.pdf(5), 0.);
    EXPECT_DOUBLE_EQ(bern.pdf(100), 0.);
}

TEST_F(bernoulli_fixture, bernoulli_pdf_always_tail)
{
    double p = 0.;
    MockVarExpr x{p};
    Bernoulli<MockVarExpr> bern = {x};
    EXPECT_DOUBLE_EQ(bern.pdf(0), 1.);
    EXPECT_DOUBLE_EQ(bern.pdf(1), 0.);
}

TEST_F(bernoulli_fixture, bernoulli_pdf_always_head)
{
    double p = 1.;
    MockVarExpr x{p};
    Bernoulli<MockVarExpr> bern = {x};
    EXPECT_DOUBLE_EQ(bern.pdf(0), 0.);
    EXPECT_DOUBLE_EQ(bern.pdf(1), 1.);
}

TEST_F(bernoulli_fixture, bernoulli_log_pdf_in_range)
{
    EXPECT_DOUBLE_EQ(bern.log_pdf(0), std::log(1-p));
    EXPECT_DOUBLE_EQ(bern.log_pdf(1), std::log(p));
}

TEST_F(bernoulli_fixture, bernoulli_log_pdf_out_of_range)
{
    EXPECT_DOUBLE_EQ(bern.log_pdf(-100), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(bern.log_pdf(-3), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(bern.log_pdf(-1), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(bern.log_pdf(2), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(bern.log_pdf(3), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(bern.log_pdf(5), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(bern.log_pdf(100), std::numeric_limits<double>::lowest());
}

TEST_F(bernoulli_fixture, bernoulli_log_pdf_always_tail)
{
    double p = 0.;
    MockVarExpr x{p};
    Bernoulli<MockVarExpr> bern = {x};
    EXPECT_DOUBLE_EQ(bern.log_pdf(0), 0.);
    EXPECT_DOUBLE_EQ(bern.log_pdf(1), std::numeric_limits<double>::lowest());
}

TEST_F(bernoulli_fixture, bernoulli_log_pdf_always_head)
{
    double p = 1.;
    MockVarExpr x{p};
    Bernoulli<MockVarExpr> bern = {x};
    EXPECT_DOUBLE_EQ(bern.log_pdf(0), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(bern.log_pdf(1), 0.);
}

TEST_F(bernoulli_fixture, bernoulli_sample) {
    std::random_device rd{};
    std::mt19937 gen{rd()};

    for (size_t i = 0; i < sample_size; i++) {
        sample[i] = bern.sample(gen);
    }

    plot_hist(sample);
}

} // namespace expr
} // namespace ppl
