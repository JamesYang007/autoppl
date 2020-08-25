#include "gtest/gtest.h"
#include <autoppl/math/density.hpp>

namespace ppl {
namespace math {

struct normal_fixture : ::testing::Test
{
protected:
    static constexpr double tol = 1e-15;
    double mean = 0.3;
    double sigma = 1.3;
    Eigen::VectorXd x_vec;
    Eigen::VectorXd mean_vec;
    Eigen::VectorXd sigma_vec;
    Eigen::MatrixXd sigma_mat;

    normal_fixture()
        : x_vec(2)
        , mean_vec(2)
        , sigma_vec(2)
        , sigma_mat(2,2)
    {
        x_vec << -1, 1;
        mean_vec << 0, 0.5;
        sigma_vec << 0.5, 0.9;
        sigma_mat << 2, 1, 1, 2;
    }
};

TEST_F(normal_fixture, pdf)
{
    EXPECT_NEAR(normal_pdf(-10.231, mean, sigma), 1.726752595588348216742E-15, tol);
    EXPECT_NEAR(normal_pdf(-5.31, mean, sigma), 2.774166877919518907166E-5, tol);
    EXPECT_DOUBLE_EQ(normal_pdf(-2.3141231, mean, sigma), 0.04063645713784323551341);
    EXPECT_DOUBLE_EQ(normal_pdf(0., mean, sigma), 0.2988151821496727914542);
    EXPECT_DOUBLE_EQ(normal_pdf(1.31, mean, sigma), 0.2269313951019926611687);
    EXPECT_DOUBLE_EQ(normal_pdf(3.21, mean, sigma), 0.02505560241243631472997);
    EXPECT_NEAR(normal_pdf(5.24551, mean, sigma), 2.20984513448306056291E-4, tol);
    EXPECT_NEAR(normal_pdf(10.5699, mean, sigma), 8.61135160183067521907E-15, tol);

    EXPECT_DOUBLE_EQ(normal_pdf(x_vec, mean, sigma), 
                     0.04941130624863308);
    EXPECT_DOUBLE_EQ(normal_pdf(x_vec, mean_vec, sigma), 
                     0.06506112407641138);
    EXPECT_DOUBLE_EQ(normal_pdf(x_vec, mean, sigma_vec), 
                     0.00889880299712749);
    EXPECT_DOUBLE_EQ(normal_pdf(x_vec, mean_vec, sigma_vec), 
                     0.04102021201209005);
    EXPECT_DOUBLE_EQ(normal_pdf(x_vec, mean, sigma_mat), 
                     0.03280470887141322);
    EXPECT_DOUBLE_EQ(normal_pdf(x_vec, mean_vec, sigma_mat), 
                     0.05127681675398978);
}

TEST_F(normal_fixture, log_pdf)
{
    EXPECT_DOUBLE_EQ(normal_log_pdf(-10.231, mean, sigma), std::log(1.726752595588348216742E-15));
    EXPECT_DOUBLE_EQ(normal_log_pdf(-5.31, mean, sigma), std::log(2.774166877919518907166E-5));
    EXPECT_DOUBLE_EQ(normal_log_pdf(-2.3141231, mean, sigma), std::log(0.04063645713784323551341));
    EXPECT_DOUBLE_EQ(normal_log_pdf(0., mean, sigma), std::log(0.2988151821496727914542));
    EXPECT_DOUBLE_EQ(normal_log_pdf(1.31, mean, sigma), std::log(0.2269313951019926611687));
    EXPECT_DOUBLE_EQ(normal_log_pdf(3.21, mean, sigma), std::log(0.02505560241243631472997));
    EXPECT_DOUBLE_EQ(normal_log_pdf(5.24551, mean, sigma), std::log(2.20984513448306056291E-4));
    EXPECT_DOUBLE_EQ(normal_log_pdf(10.5699, mean, sigma), std::log(8.61135160183067521907E-15));

    EXPECT_DOUBLE_EQ(normal_log_pdf(x_vec, mean, sigma), 
                     -3.007576009545511);
    EXPECT_DOUBLE_EQ(normal_log_pdf(x_vec, mean_vec, sigma), 
                     -2.7324280805514283);
    EXPECT_DOUBLE_EQ(normal_log_pdf(x_vec, mean, sigma_vec), 
                     -4.721838505994043);
    EXPECT_DOUBLE_EQ(normal_log_pdf(x_vec, mean_vec, sigma_vec), 
                     -3.1936903578458944);
    EXPECT_DOUBLE_EQ(normal_log_pdf(x_vec, mean, sigma_mat), 
                     -3.4171832107434);
    EXPECT_DOUBLE_EQ(normal_log_pdf(x_vec, mean_vec, sigma_mat), 
                     -2.970516544076734);
}

struct uniform_fixture : ::testing::Test
{
protected:
    double min = -2.3;
    double max = 2.7;
    Eigen::VectorXd x_vec;
    Eigen::VectorXd min_vec;
    Eigen::VectorXd max_vec;

    uniform_fixture()
        : x_vec(2)
        , min_vec(2)
        , max_vec(2)
    {
        x_vec << 0, -1;
        min_vec << -1, -2;
        max_vec << 0.1, 0;
    }
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

    EXPECT_DOUBLE_EQ(uniform_pdf(x_vec, min, max), 0.2 * 0.2);
    EXPECT_DOUBLE_EQ(uniform_pdf(x_vec, min, max_vec), 
                     1./((max_vec(0) - min) * (max_vec(1) - min)));
    EXPECT_DOUBLE_EQ(uniform_pdf(x_vec, min_vec, max), 
                     1./((max - min_vec(0)) * (max - min_vec(1))));
    EXPECT_DOUBLE_EQ(uniform_pdf(x_vec, min_vec, max_vec), 
                     1./((max_vec(0) - min_vec(0)) * (max_vec(1) - min_vec(1))));
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

    EXPECT_DOUBLE_EQ(uniform_log_pdf(x_vec, min, max), 2*std::log(0.2));
    EXPECT_DOUBLE_EQ(uniform_log_pdf(x_vec, min, max_vec), 
                     -std::log(max_vec(0) - min) - std::log(max_vec(1) - min));
    EXPECT_DOUBLE_EQ(uniform_log_pdf(x_vec, min_vec, max), 
                     -std::log(max - min_vec(0)) - std::log(max - min_vec(1)));
    EXPECT_DOUBLE_EQ(uniform_log_pdf(x_vec, min_vec, max_vec), 
                     -std::log(max_vec(0) - min_vec(0)) -std::log(max_vec(1) - min_vec(1)));
}

TEST_F(uniform_fixture, uniform_log_pdf_out_of_range)
{
    EXPECT_DOUBLE_EQ(uniform_log_pdf(-100., min, max), neg_inf<double>);
    EXPECT_DOUBLE_EQ(uniform_log_pdf(-3.41, min, max), neg_inf<double>);
    EXPECT_DOUBLE_EQ(uniform_log_pdf(-2.3, min, max), neg_inf<double>);
    EXPECT_DOUBLE_EQ(uniform_log_pdf(2.7, min, max), neg_inf<double>);
    EXPECT_DOUBLE_EQ(uniform_log_pdf(3.5, min, max), neg_inf<double>);
    EXPECT_DOUBLE_EQ(uniform_log_pdf(3214., min, max), neg_inf<double>);

    x_vec(0) = 1000;
    EXPECT_DOUBLE_EQ(uniform_log_pdf(x_vec, min_vec, max_vec), neg_inf<double>);
}

struct bernoulli_fixture : ::testing::Test 
{
protected:
    double p = 0.6;
    Eigen::VectorXd x_vec;
    Eigen::VectorXd p_vec;

    bernoulli_fixture()
        : x_vec(3)
        , p_vec(3)
    {
        x_vec << 1, 0, 1;
        p_vec << 0.3, 0.5, 0.1;
    }
};

TEST_F(bernoulli_fixture, bernoulli_pdf_in_range)
{
    EXPECT_DOUBLE_EQ(bernoulli_pdf(0, p), 1-p);
    EXPECT_DOUBLE_EQ(bernoulli_pdf(1, p), p);
    EXPECT_DOUBLE_EQ(bernoulli_pdf(x_vec, p), p*p*(1-p));
    EXPECT_DOUBLE_EQ(bernoulli_pdf(x_vec, p_vec), 
                     p_vec(0) * (1-p_vec(1)) * p_vec(2));
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
    EXPECT_DOUBLE_EQ(bernoulli_pdf(0, -0.1), 1.);
    EXPECT_DOUBLE_EQ(bernoulli_pdf(0, 1.3), 0.);
    EXPECT_DOUBLE_EQ(bernoulli_pdf(1, -0.1), 0.);
    EXPECT_DOUBLE_EQ(bernoulli_pdf(1, 1.3), 1.);

    p_vec(1) = 1.23;
    EXPECT_DOUBLE_EQ(bernoulli_pdf(x_vec, p_vec), 0);
    p_vec(1) = -0.3;
    EXPECT_DOUBLE_EQ(bernoulli_pdf(x_vec, p_vec), p_vec(0) * p_vec(2));
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
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(0, -0.1), 0);
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(0, 1.3), neg_inf<double>);
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(1, -0.1), neg_inf<double>);
    EXPECT_DOUBLE_EQ(bernoulli_log_pdf(1, 1.3), 0);
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
