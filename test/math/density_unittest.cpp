#include "gtest/gtest.h"
#include <autoppl/math/density.hpp>

namespace ppl {
namespace math {

struct normal_fixture : ::testing::Test
{
protected:
    static constexpr double tol = 1e-15;
    double mean = 0.3;
    double sd = 1.3;
};

TEST_F(normal_fixture, pdf)
{
    EXPECT_NEAR(normal_pdf(-10.231, mean, sd), 1.726752595588348216742E-15, tol);
    EXPECT_NEAR(normal_pdf(-5.31, mean, sd), 2.774166877919518907166E-5, tol);
    EXPECT_DOUBLE_EQ(normal_pdf(-2.3141231, mean, sd), 0.04063645713784323551341);
    EXPECT_DOUBLE_EQ(normal_pdf(0., mean, sd), 0.2988151821496727914542);
    EXPECT_DOUBLE_EQ(normal_pdf(1.31, mean, sd), 0.2269313951019926611687);
    EXPECT_DOUBLE_EQ(normal_pdf(3.21, mean, sd), 0.02505560241243631472997);
    EXPECT_NEAR(normal_pdf(5.24551, mean, sd), 2.20984513448306056291E-4, tol);
    EXPECT_NEAR(normal_pdf(10.5699, mean, sd), 8.61135160183067521907E-15, tol);
}

TEST_F(normal_fixture, log_pdf)
{
    EXPECT_DOUBLE_EQ(normal_log_pdf(-10.231, mean, sd), std::log(1.726752595588348216742E-15));
    EXPECT_DOUBLE_EQ(normal_log_pdf(-5.31, mean, sd), std::log(2.774166877919518907166E-5));
    EXPECT_DOUBLE_EQ(normal_log_pdf(-2.3141231, mean, sd), std::log(0.04063645713784323551341));
    EXPECT_DOUBLE_EQ(normal_log_pdf(0., mean, sd), std::log(0.2988151821496727914542));
    EXPECT_DOUBLE_EQ(normal_log_pdf(1.31, mean, sd), std::log(0.2269313951019926611687));
    EXPECT_DOUBLE_EQ(normal_log_pdf(3.21, mean, sd), std::log(0.02505560241243631472997));
    EXPECT_DOUBLE_EQ(normal_log_pdf(5.24551, mean, sd), std::log(2.20984513448306056291E-4));
    EXPECT_DOUBLE_EQ(normal_log_pdf(10.5699, mean, sd), std::log(8.61135160183067521907E-15));
}

struct uniform_fixture : ::testing::Test
{
protected:
    double min = -2.3;
    double max = 2.7;
};

TEST_F(uniform_fixture, uniform_pdf_in_range)
{
    EXPECT_DOUBLE_EQ(uniform_pdf(-2.2999999999, min, max), 0.2);
    EXPECT_DOUBLE_EQ(uniform_pdf(-2., min, max), 0.2);
    EXPECT_DOUBLE_EQ(uniform_pdf(-1.423, min, max), 0.2);
    EXPECT_DOUBLE_EQ(uniform_pdf(0., min, max), 0.2);
    EXPECT_DOUBLE_EQ(uniform_pdf(1.31, min, max), 0.2);
    EXPECT_DOUBLE_EQ(uniform_pdf(2.41, min, max), 0.2);
    EXPECT_DOUBLE_EQ(uniform_pdf(2.69999999999, min, max), 0.2);
}

TEST_F(uniform_fixture, uniform_pdf_out_of_range)
{
    EXPECT_DOUBLE_EQ(uniform_pdf(-100., min, max), 0.);
    EXPECT_DOUBLE_EQ(uniform_pdf(-3.41, min, max), 0.);
    EXPECT_DOUBLE_EQ(uniform_pdf(-2.3, min, max), 0.);
    EXPECT_DOUBLE_EQ(uniform_pdf(2.7, min, max), 0.);
    EXPECT_DOUBLE_EQ(uniform_pdf(3.5, min, max), 0.);
    EXPECT_DOUBLE_EQ(uniform_pdf(3214., min, max), 0.);
}

TEST_F(uniform_fixture, uniform_log_pdf_in_range)
{
    EXPECT_DOUBLE_EQ(uniform_log_pdf(-2.2999999999, min, max), std::log(0.2));
    EXPECT_DOUBLE_EQ(uniform_log_pdf(-2., min, max), std::log(0.2));
    EXPECT_DOUBLE_EQ(uniform_log_pdf(-1.423, min, max), std::log(0.2));
    EXPECT_DOUBLE_EQ(uniform_log_pdf(0., min, max), std::log(0.2));
    EXPECT_DOUBLE_EQ(uniform_log_pdf(1.31, min, max), std::log(0.2));
    EXPECT_DOUBLE_EQ(uniform_log_pdf(2.41, min, max), std::log(0.2));
    EXPECT_DOUBLE_EQ(uniform_log_pdf(2.69999999999, min, max), std::log(0.2));
}

TEST_F(uniform_fixture, uniform_log_pdf_out_of_range)
{
    EXPECT_DOUBLE_EQ(uniform_log_pdf(-100., min, max), neg_inf<double>);
    EXPECT_DOUBLE_EQ(uniform_log_pdf(-3.41, min, max), neg_inf<double>);
    EXPECT_DOUBLE_EQ(uniform_log_pdf(-2.3, min, max), neg_inf<double>);
    EXPECT_DOUBLE_EQ(uniform_log_pdf(2.7, min, max), neg_inf<double>);
    EXPECT_DOUBLE_EQ(uniform_log_pdf(3.5, min, max), neg_inf<double>);
    EXPECT_DOUBLE_EQ(uniform_log_pdf(3214., min, max), neg_inf<double>);
}

struct bernoulli_fixture : ::testing::Test 
{
protected:
    double p = 0.6;
};

TEST_F(bernoulli_fixture, bernoulli_pdf_in_range)
{
    EXPECT_DOUBLE_EQ(bernoulli_pdf(0, p), 1-p);
    EXPECT_DOUBLE_EQ(bernoulli_pdf(1, p), p);
}

TEST_F(bernoulli_fixture, bernoulli_pdf_out_of_range)
{
    EXPECT_DOUBLE_EQ(bernoulli_pdf(-100, p), 0.);
    EXPECT_DOUBLE_EQ(bernoulli_pdf(-3, p), 0.);
    EXPECT_DOUBLE_EQ(bernoulli_pdf(-2, p), 0.);
    EXPECT_DOUBLE_EQ(bernoulli_pdf(2, p), 0.);
    EXPECT_DOUBLE_EQ(bernoulli_pdf(3, p), 0.);
    EXPECT_DOUBLE_EQ(bernoulli_pdf(5, p), 0.);
    EXPECT_DOUBLE_EQ(bernoulli_pdf(100, p), 0.);
}

TEST_F(bernoulli_fixture, bernoulli_pdf_always_tail)
{
    double p = 0.;
    EXPECT_DOUBLE_EQ(bernoulli_pdf(0, p), 1.);
    EXPECT_DOUBLE_EQ(bernoulli_pdf(1, p), 0.);
}

TEST_F(bernoulli_fixture, bernoulli_pdf_always_head)
{
    double p = 1.;
    EXPECT_DOUBLE_EQ(bernoulli_pdf(0, p), 0.);
    EXPECT_DOUBLE_EQ(bernoulli_pdf(1, p), 1.);
}

TEST_F(bernoulli_fixture, bernoulli_log_pdf_in_range)
{
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(0, p), std::log(1-p));
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(1, p), std::log(p));
}

TEST_F(bernoulli_fixture, bernoulli_log_pdf_out_of_range)
{
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(-100, p), neg_inf<double>);
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(-3, p), neg_inf<double>);
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(-1, p), neg_inf<double>);
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(2, p), neg_inf<double>);
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(3, p), neg_inf<double>);
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(5, p), neg_inf<double>);
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(100, p), neg_inf<double>);
}

TEST_F(bernoulli_fixture, bernoulli_log_pdf_always_tail)
{
    double p = 0.;
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(0, p), 0.);
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(1, p), neg_inf<double>);
}

TEST_F(bernoulli_fixture, bernoulli_log_pdf_always_head)
{
    double p = 1.;
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(0, p), neg_inf<double>);
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(1, p), 0.);
}
    

} // namespace math
} // namespace ppl
