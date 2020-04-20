#include "gtest/gtest.h"
#include <array>
#include <autoppl/expression/distribution/density.hpp>

namespace ppl {
namespace expr {

struct density_fixture : ::testing::Test
{
protected:
    double x = 0.;
    double min = -2.3;
    double max = 2.7;
    double mean = 0.3;
    double stddev = 1.3;
    double tol = 1e-15;
    double p = 0.41;
};

/*
 * Continuous distribution
 */

TEST_F(density_fixture, uniform_pdf_in_range)
{
    EXPECT_DOUBLE_EQ(UniformBase::pdf(-2.2999999999, min, max), 0.2);
    EXPECT_DOUBLE_EQ(UniformBase::pdf(-2., min, max), 0.2);
    EXPECT_DOUBLE_EQ(UniformBase::pdf(-1.423, min, max), 0.2);
    EXPECT_DOUBLE_EQ(UniformBase::pdf(0., min, max), 0.2);
    EXPECT_DOUBLE_EQ(UniformBase::pdf(1.31, min, max), 0.2);
    EXPECT_DOUBLE_EQ(UniformBase::pdf(2.41, min, max), 0.2);
    EXPECT_DOUBLE_EQ(UniformBase::pdf(2.69999999999, min, max), 0.2);
}

TEST_F(density_fixture, uniform_pdf_out_of_range)
{
    EXPECT_DOUBLE_EQ(UniformBase::pdf(-100, min, max), 0.);
    EXPECT_DOUBLE_EQ(UniformBase::pdf(-3.41, min, max), 0.);
    EXPECT_DOUBLE_EQ(UniformBase::pdf(-2.3, min, max), 0.);
    EXPECT_DOUBLE_EQ(UniformBase::pdf(2.7, min, max), 0.);
    EXPECT_DOUBLE_EQ(UniformBase::pdf(3.5, min, max), 0.);
    EXPECT_DOUBLE_EQ(UniformBase::pdf(3214, min, max), 0.);
}

TEST_F(density_fixture, uniform_log_pdf_in_range)
{
    EXPECT_DOUBLE_EQ(UniformBase::log_pdf(-2.2999999999, min, max), std::log(0.2));
    EXPECT_DOUBLE_EQ(UniformBase::log_pdf(-2., min, max), std::log(0.2));
    EXPECT_DOUBLE_EQ(UniformBase::log_pdf(-1.423, min, max), std::log(0.2));
    EXPECT_DOUBLE_EQ(UniformBase::log_pdf(0., min, max), std::log(0.2));
    EXPECT_DOUBLE_EQ(UniformBase::log_pdf(1.31, min, max), std::log(0.2));
    EXPECT_DOUBLE_EQ(UniformBase::log_pdf(2.41, min, max), std::log(0.2));
    EXPECT_DOUBLE_EQ(UniformBase::log_pdf(2.69999999999, min, max), std::log(0.2));
}

TEST_F(density_fixture, uniform_log_pdf_out_of_range)
{
    EXPECT_DOUBLE_EQ(UniformBase::log_pdf(-100, min, max), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(UniformBase::log_pdf(-3.41, min, max), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(UniformBase::log_pdf(-2.3, min, max), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(UniformBase::log_pdf(2.7, min, max), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(UniformBase::log_pdf(3.5, min, max), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(UniformBase::log_pdf(3214, min, max), std::numeric_limits<double>::lowest());
}

TEST_F(density_fixture, normal_pdf)
{
    EXPECT_NEAR(NormalBase::pdf(-10.231, mean, stddev), 1.726752595588348216742E-15, tol);
    EXPECT_NEAR(NormalBase::pdf(-5.31, mean, stddev), 2.774166877919518907166E-5, tol);
    EXPECT_DOUBLE_EQ(NormalBase::pdf(-2.3141231, mean, stddev), 0.04063645713784323551341);
    EXPECT_DOUBLE_EQ(NormalBase::pdf(0., mean, stddev), 0.2988151821496727914542);
    EXPECT_DOUBLE_EQ(NormalBase::pdf(1.31, mean, stddev), 0.2269313951019926611687);
    EXPECT_DOUBLE_EQ(NormalBase::pdf(3.21, mean, stddev), 0.02505560241243631472997);
    EXPECT_NEAR(NormalBase::pdf(5.24551, mean, stddev), 2.20984513448306056291E-4, tol);
    EXPECT_NEAR(NormalBase::pdf(10.5699, mean, stddev), 8.61135160183067521907E-15, tol);
}

TEST_F(density_fixture, normal_log_pdf)
{
    EXPECT_DOUBLE_EQ(NormalBase::log_pdf(-10.231, mean, stddev), std::log(1.726752595588348216742E-15));
    EXPECT_DOUBLE_EQ(NormalBase::log_pdf(-5.31, mean, stddev), std::log(2.774166877919518907166E-5));
    EXPECT_DOUBLE_EQ(NormalBase::log_pdf(-2.3141231, mean, stddev), std::log(0.04063645713784323551341));
    EXPECT_DOUBLE_EQ(NormalBase::log_pdf(0., mean, stddev), std::log(0.2988151821496727914542));
    EXPECT_DOUBLE_EQ(NormalBase::log_pdf(1.31, mean, stddev), std::log(0.2269313951019926611687));
    EXPECT_DOUBLE_EQ(NormalBase::log_pdf(3.21, mean, stddev), std::log(0.02505560241243631472997));
    EXPECT_DOUBLE_EQ(NormalBase::log_pdf(5.24551, mean, stddev), std::log(2.20984513448306056291E-4));
    EXPECT_DOUBLE_EQ(NormalBase::log_pdf(10.5699, mean, stddev), std::log(8.61135160183067521907E-15));
}

/*
 * Discrete distributions
 */

TEST_F(density_fixture, bernoulli_pdf_in_range)
{
    EXPECT_DOUBLE_EQ(BernoulliBase::pdf(0, p), 1-p);
    EXPECT_DOUBLE_EQ(BernoulliBase::pdf(1, p), p);
}

TEST_F(density_fixture, bernoulli_pdf_out_of_range)
{
    EXPECT_DOUBLE_EQ(BernoulliBase::pdf(-100, p), 0.);
    EXPECT_DOUBLE_EQ(BernoulliBase::pdf(-3.41, p), 0.);
    EXPECT_DOUBLE_EQ(BernoulliBase::pdf(-0.00000001, p), 0.);
    EXPECT_DOUBLE_EQ(BernoulliBase::pdf(0.00000001, p), 0.);
    EXPECT_DOUBLE_EQ(BernoulliBase::pdf(0.99999999, p), 0.);
    EXPECT_DOUBLE_EQ(BernoulliBase::pdf(1.00000001, p), 0.);
    EXPECT_DOUBLE_EQ(BernoulliBase::pdf(3.1423, p), 0.);
    EXPECT_DOUBLE_EQ(BernoulliBase::pdf(5.613, p), 0.);
    EXPECT_DOUBLE_EQ(BernoulliBase::pdf(100., p), 0.);
}

TEST_F(density_fixture, bernoulli_pdf_always_tail)
{
    EXPECT_DOUBLE_EQ(BernoulliBase::pdf(0, 0.), 1.);
    EXPECT_DOUBLE_EQ(BernoulliBase::pdf(1, 0.), 0.);
}

TEST_F(density_fixture, bernoulli_pdf_always_head)
{
    EXPECT_DOUBLE_EQ(BernoulliBase::pdf(0, 1.), 0.);
    EXPECT_DOUBLE_EQ(BernoulliBase::pdf(1, 1.), 1.);
}

TEST_F(density_fixture, bernoulli_log_pdf_in_range)
{
    EXPECT_DOUBLE_EQ(BernoulliBase::log_pdf(0, p), std::log(1-p));
    EXPECT_DOUBLE_EQ(BernoulliBase::log_pdf(1, p), std::log(p));
}

TEST_F(density_fixture, bernoulli_log_pdf_out_of_range)
{
    EXPECT_DOUBLE_EQ(BernoulliBase::log_pdf(-100, p), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(BernoulliBase::log_pdf(-3.41, p), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(BernoulliBase::log_pdf(-0.00000001, p), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(BernoulliBase::log_pdf(0.00000001, p), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(BernoulliBase::log_pdf(0.99999999, p), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(BernoulliBase::log_pdf(1.00000001, p), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(BernoulliBase::log_pdf(3.1423, p), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(BernoulliBase::log_pdf(5.613, p), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(BernoulliBase::log_pdf(100., p), std::numeric_limits<double>::lowest());
}

TEST_F(density_fixture, bernoulli_log_pdf_always_tail)
{
    EXPECT_DOUBLE_EQ(BernoulliBase::log_pdf(0, 0.), 0.);
    EXPECT_DOUBLE_EQ(BernoulliBase::log_pdf(1, 0.), std::numeric_limits<double>::lowest());
}

TEST_F(density_fixture, bernoulli_log_pdf_always_head)
{
    EXPECT_DOUBLE_EQ(BernoulliBase::log_pdf(0, 1.), std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(BernoulliBase::log_pdf(1, 1.), 0.);
}

} // namespace expr
} // namespace ppl
