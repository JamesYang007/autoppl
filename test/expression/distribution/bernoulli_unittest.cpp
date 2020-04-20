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
    static_assert(util::is_dist_expr_v<Bernoulli<MockVarExpr>>);
}

TEST_F(bernoulli_fixture, bernoulli_check_params) {
    EXPECT_DOUBLE_EQ(bern.p(), static_cast<value_t>(x));
}

TEST_F(bernoulli_fixture, bernoulli_pdf_delegate) {
    EXPECT_DOUBLE_EQ(bern.pdf(-10), BernoulliBase::pdf(-10, p));
    EXPECT_DOUBLE_EQ(bern.pdf(-7), BernoulliBase::pdf(-7, p));
    EXPECT_DOUBLE_EQ(bern.pdf(-3), BernoulliBase::pdf(-3, p));
    EXPECT_DOUBLE_EQ(bern.pdf(0), BernoulliBase::pdf(0, p));
    EXPECT_DOUBLE_EQ(bern.pdf(1), BernoulliBase::pdf(1, p));
    EXPECT_DOUBLE_EQ(bern.pdf(3), BernoulliBase::pdf(3, p));
    EXPECT_DOUBLE_EQ(bern.pdf(6), BernoulliBase::pdf(6, p));
    EXPECT_DOUBLE_EQ(bern.pdf(16), BernoulliBase::pdf(16, p));
}

TEST_F(bernoulli_fixture, bernoulli_log_pdf_delegate) {
    EXPECT_DOUBLE_EQ(bern.log_pdf(-10), BernoulliBase::log_pdf(-10, p));
    EXPECT_DOUBLE_EQ(bern.log_pdf(-7), BernoulliBase::log_pdf(-7, p));
    EXPECT_DOUBLE_EQ(bern.log_pdf(-3), BernoulliBase::log_pdf(-3, p));
    EXPECT_DOUBLE_EQ(bern.log_pdf(0), BernoulliBase::log_pdf(0, p));
    EXPECT_DOUBLE_EQ(bern.log_pdf(1), BernoulliBase::log_pdf(1, p));
    EXPECT_DOUBLE_EQ(bern.log_pdf(3), BernoulliBase::log_pdf(3, p));
    EXPECT_DOUBLE_EQ(bern.log_pdf(6), BernoulliBase::log_pdf(6, p));
    EXPECT_DOUBLE_EQ(bern.log_pdf(16), BernoulliBase::log_pdf(16, p));
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
