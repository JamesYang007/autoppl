#include "gtest/gtest.h"
#include <array>
#include <autoppl/expr_builder.hpp>

namespace ppl {

struct ad_integration_fixture : ::testing::Test
{
protected:
    Data<double> x{1., 2., 3.}, y{0., -1., 1.};
    Param<double> theta;
    std::array<const void*, 1> keys = {&theta};
    std::vector<ad::Var<double>> vars;

    ad_integration_fixture()
        : theta{}
        , vars(1)
    {
        vars[0].set_value(1.);
    }
};

TEST_F(ad_integration_fixture, ad_log_pdf_data_constant_param)
{
    auto model = (x |= normal(0., 1.));
    auto ad_expr = model.ad_log_pdf(keys, vars);
    double value = ad::evaluate(ad_expr);
    EXPECT_DOUBLE_EQ(value, -0.5 * 14);
    value = ad::autodiff(ad_expr); // should not affect the result
    EXPECT_DOUBLE_EQ(value, -0.5 * 14);
}

TEST_F(ad_integration_fixture, ad_log_pdf_data_mean_param)
{
    auto model = (
        theta |= normal(0., 2.),
        x |= normal(theta, 1.)
    );
    auto ad_expr = model.ad_log_pdf(keys, vars);

    double value = ad::autodiff(ad_expr);
    EXPECT_DOUBLE_EQ(value, -0.5 * 5 - 1./8 - std::log(2));
    EXPECT_DOUBLE_EQ(vars[0].get_adjoint(), 2.75);

    // after resetting adjoint, differentiating should not change anything
    vars[0].reset_adjoint();

    value = ad::autodiff(ad_expr);
    EXPECT_DOUBLE_EQ(value, -0.5 * 5 - 1./8 - std::log(2));
    EXPECT_DOUBLE_EQ(vars[0].get_adjoint(), 2.75);
}

TEST_F(ad_integration_fixture, ad_log_pdf_data_stddev_param)
{
    auto model = (
        theta |= normal(0., 2.),
        x |= normal(0., theta)
    );

    auto ad_expr = model.ad_log_pdf(keys, vars);

    double value = ad::autodiff(ad_expr);
    EXPECT_DOUBLE_EQ(value, -0.5 * 14 - 1./8 - std::log(2));
    EXPECT_DOUBLE_EQ(vars[0].get_adjoint(), 10.75);

    // after resetting adjoint, differentiating should not change anything
    vars[0].reset_adjoint();

    value = ad::autodiff(ad_expr);
    EXPECT_DOUBLE_EQ(value, -0.5 * 14 - 1./8 - std::log(2));
    EXPECT_DOUBLE_EQ(vars[0].get_adjoint(), 10.75);
}

TEST_F(ad_integration_fixture, ad_log_pdf_data_param_with_data)
{
    auto model = (
        theta |= normal(0., 1.),
        y |= normal(theta * x, 1.)
    );

    auto ad_expr = model.ad_log_pdf(keys, vars);

    double value = ad::autodiff(ad_expr);
    EXPECT_DOUBLE_EQ(value, -7.5);
    EXPECT_DOUBLE_EQ(vars[0].get_adjoint(), -14.);

    // after resetting adjoint, differentiating should not change anything
    vars[0].reset_adjoint();

    value = ad::autodiff(ad_expr);
    EXPECT_DOUBLE_EQ(value, -7.5);
    EXPECT_DOUBLE_EQ(vars[0].get_adjoint(), -14.);
}

} // namespace ppl
