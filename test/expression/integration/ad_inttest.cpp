#include "gtest/gtest.h"
#include <array>
#include <autoppl/expression/expr_builder.hpp>
#include <autoppl/expression/activate.hpp>

namespace ppl {

struct ad_integration_fixture : ::testing::Test
{
protected:
    using value_t = double;
    using data_t = Data<value_t, ppl::vec>;
    using param_t = Param<value_t, ppl::scl>;
    using pview_t = ParamView<
        typename util::param_traits<param_t>::pointer_t, 
        ppl::scl>;

    data_t x{1., 2., 3.}, y{0., -1., 1.};
    param_t theta;
    std::vector<ad::Var<value_t>> vars;
    std::vector<ad::Var<value_t>> cache;

    ad_integration_fixture()
        : theta{}
        , vars(1)
        , cache(100)  // obscene amount of cache
    {
        vars[0].set_value(1.);
    }

    template <class ADVecType>
    void reset_cache(ADVecType& c)
    {
        for (auto& x : c) { x.reset_adjoint(); }
    }
};

TEST_F(ad_integration_fixture, ad_log_pdf_data_constant_param)
{
    auto model = (x |= normal(0., 1.));
    expr::activate(model);
    expr::activate_cache(model);

    auto ad_expr = model.ad_log_pdf(vars, cache);
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
    expr::activate(model);
    expr::activate_cache(model);

    auto ad_expr = model.ad_log_pdf(vars, cache);

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
    expr::activate(model);
    expr::activate_cache(model);

    auto ad_expr = model.ad_log_pdf(vars, cache);

    double value = ad::autodiff(ad_expr);
    EXPECT_DOUBLE_EQ(value, -0.5 * 14 - 1./8 - std::log(2));
    EXPECT_DOUBLE_EQ(vars[0].get_adjoint(), 10.75);

    // after resetting adjoint and cache, 
    // differentiating should not change anything
    vars[0].reset_adjoint();
    reset_cache(cache);

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
    expr::activate(model);
    expr::activate_cache(model);

    auto ad_expr = model.ad_log_pdf(vars, cache);

    double value = ad::autodiff(ad_expr);
    EXPECT_DOUBLE_EQ(value, -7.5);
    EXPECT_DOUBLE_EQ(vars[0].get_adjoint(), -14.);

    // after resetting adjoint, differentiating should not change anything
    vars[0].reset_adjoint();

    value = ad::autodiff(ad_expr);
    EXPECT_DOUBLE_EQ(value, -7.5);
    EXPECT_DOUBLE_EQ(vars[0].get_adjoint(), -14.);
}

TEST_F(ad_integration_fixture, ad_log_pdf_constant_param_within_bounds)
{
    auto model = (
        theta |= uniform(-1., 0.5)
    );
    expr::activate(model);
    expr::activate_cache(model);

    auto expr = model.ad_log_pdf(vars, cache);
    double value = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(value, math::neg_inf<double>);
    EXPECT_DOUBLE_EQ(vars[0].get_adjoint(), 0);
}

TEST_F(ad_integration_fixture, ad_log_pdf_constant_param_out_of_bounds)
{
    vars[0].set_value(0.4999);
    auto model = (
        theta |= uniform(-1., 0.5)
    );
    expr::activate(model);
    expr::activate_cache(model);

    auto expr = model.ad_log_pdf(vars, cache);
    double value = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(value, -std::log(1.5));
    EXPECT_DOUBLE_EQ(vars[0].get_adjoint(), 0);
}

TEST_F(ad_integration_fixture, ad_log_pdf_var_param_within_bounds)
{
    vars[0].set_value(0.42);
    auto model = (
        theta |= normal(-1., 0.5),
        x |= uniform(theta, theta + 5.)
    );
    expr::activate(model);
    expr::activate_cache(model);

    auto expr = model.ad_log_pdf(vars, cache);
    double value = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(value, -2*(1.42 * 1.42) + std::log(2) - 3*std::log(5));
}

TEST_F(ad_integration_fixture, ad_log_pdf_var_param_out_of_bounds)
{
    vars[0].set_value(0.42);
    auto model = (
        theta |= normal(-1., 0.5),
        x |= uniform(theta, theta + 2)
    );
    expr::activate(model);
    expr::activate_cache(model);

    auto expr = model.ad_log_pdf(vars, cache);
    double value = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(value, math::neg_inf<double>);
}

} // namespace ppl
