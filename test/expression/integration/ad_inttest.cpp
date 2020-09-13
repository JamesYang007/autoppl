#include "gtest/gtest.h"
#include <array>
#include <fastad>
#include <testutil/base_fixture.hpp>
#include <autoppl/expression/model/bar_eq.hpp>
#include <autoppl/expression/model/glue.hpp>
#include <autoppl/expression/constraint/bounded.hpp>
#include <autoppl/expression/constraint/lower.hpp>
#include <autoppl/expression/constraint/pos_def.hpp>
#include <autoppl/expression/variable/binary.hpp>
#include <autoppl/expression/variable/constant.hpp>
#include <autoppl/expression/variable/for_each.hpp>
#include <autoppl/expression/variable/op_eq.hpp>
#include <autoppl/expression/variable/glue.hpp>
#include <autoppl/util/ad_boost/bounded_inv_transform.hpp>
#include <autoppl/util/ad_boost/lower_inv_transform.hpp>
#include <autoppl/util/ad_boost/cov_inv_transform.hpp>
#include <autoppl/util/iterator/counting_iterator.hpp>
#include <autoppl/expression/distribution/normal.hpp>
#include <autoppl/expression/distribution/uniform.hpp>
#include <autoppl/expression/distribution/cauchy.hpp>
#include <autoppl/expression/distribution/wishart.hpp>
#include <autoppl/expression/program/activate.hpp>
#include <autoppl/expression/op_overloads.hpp>

namespace ppl {

struct ad_integration_fixture: 
    base_fixture<double>,
    ::testing::Test
{
protected:
    using data_t = vec_d_t;
    using param_t = scl_p_t;
    using pview_t = scl_pv_t;

    data_t x, y;
    param_t theta;
    Eigen::VectorXd vals;
    Eigen::VectorXd adjs;

    value_t tol = 2e-15;

    ad_integration_fixture()
        : x(3)
        , y(3)
        , theta{}
        , vals(1)
        , adjs(1) 
    {
        x.get() << 1., 2., 3.;
        y.get() << 0., -1., 1.;
        vals(0) = 1.;
        adjs(0) = 0.;

        ptr_pack.uc_val = vals.data();
        ptr_pack.uc_adj = adjs.data();
    }
};

TEST_F(ad_integration_fixture, ad_log_pdf_data_constant_param)
{
    auto model = (x |= normal(0., 1.));
    expr::activate(model);

    auto expr = ad::bind(model.ad_log_pdf(ptr_pack));
    double value = ad::evaluate(expr);
    EXPECT_DOUBLE_EQ(value, -0.5 * 14);
    value = ad::autodiff(expr); // should not affect the result
    EXPECT_DOUBLE_EQ(value, -0.5 * 14);
}

TEST_F(ad_integration_fixture, ad_log_pdf_data_mean_param)
{
    auto model = (
        theta |= normal(0., 2.),
        x |= normal(theta, 1.)
    );
    expr::activate(model);

    auto expr = ad::bind(model.ad_log_pdf(ptr_pack));

    double value = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(value, -0.5 * 5 - 1./8 - std::log(2));
    EXPECT_DOUBLE_EQ(adjs[0], 2.75);

    // after resetting adjoint, differentiating should not change anything
    adjs.setZero();

    value = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(value, -0.5 * 5 - 1./8 - std::log(2));
    EXPECT_DOUBLE_EQ(adjs(0), 2.75);
}

TEST_F(ad_integration_fixture, ad_log_pdf_data_stddev_param)
{
    auto model = (
        theta |= normal(0., 2.),
        x |= normal(0., theta)
    );
    expr::activate(model);

    auto expr = ad::bind(model.ad_log_pdf(ptr_pack));

    double value = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(value, -0.5 * 14 - 1./8 - std::log(2));
    EXPECT_DOUBLE_EQ(adjs(0), 10.75);

    // after resetting adjoint and cache, 
    // differentiating should not change anything
    adjs.setZero();

    value = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(value, -0.5 * 14 - 1./8 - std::log(2));
    EXPECT_DOUBLE_EQ(adjs(0), 10.75);
}

TEST_F(ad_integration_fixture, ad_log_pdf_data_param_with_data)
{
    auto model = (
        theta |= normal(0., 1.),
        y |= normal(theta * x, 1.)
    );
    expr::activate(model);

    auto expr = ad::bind(model.ad_log_pdf(ptr_pack));

    double value = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(value, -7.5);
    EXPECT_DOUBLE_EQ(adjs(0), -14.);

    // after resetting adjoint, differentiating should not change anything
    adjs.setZero();

    value = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(value, -7.5);
    EXPECT_DOUBLE_EQ(adjs(0), -14.);
}

TEST_F(ad_integration_fixture, ad_log_pdf_constant_param_within_bounds)
{
    auto model = (
        theta |= uniform(-1., 0.5)
    );
    expr::activate(model);

    auto expr = ad::bind(model.ad_log_pdf(ptr_pack));
    double value = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(value, math::neg_inf<double>);
    EXPECT_DOUBLE_EQ(adjs(0), 0);
}

TEST_F(ad_integration_fixture, ad_log_pdf_constant_param_out_of_bounds)
{
    vals(0) = 0.4999;
    auto model = (
        theta |= uniform(-1., 0.5)
    );
    expr::activate(model);

    auto expr = ad::bind(model.ad_log_pdf(ptr_pack));
    double value = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(value, -std::log(1.5));
    EXPECT_DOUBLE_EQ(adjs(0), 0);
}

TEST_F(ad_integration_fixture, ad_log_pdf_var_param_within_bounds)
{
    vals(0) = 0.42;
    auto model = (
        theta |= normal(-1., 0.5),
        x |= uniform(theta, theta + 5.)
    );
    expr::activate(model);

    auto expr = ad::bind(model.ad_log_pdf(ptr_pack));
    double value = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(value, -2*(1.42 * 1.42) + std::log(2) - 3*std::log(5));
}

TEST_F(ad_integration_fixture, ad_log_pdf_var_param_out_of_bounds)
{
    vals(0) = 0.42;
    auto model = (
        theta |= normal(-1., 0.5),
        x |= uniform(theta, theta + 2)
    );
    expr::activate(model);

    auto expr = ad::bind(model.ad_log_pdf(ptr_pack));
    double value = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(value, math::neg_inf<double>);
}

TEST_F(ad_integration_fixture, ad_log_pdf_pos_def)
{
    vec_d_t y(2);
    y.get() << 1., -1.;
    Eigen::MatrixXd V(2,2);
    V << 2, 1, 1, 2;
    Param Sigma = make_param<double, mat>(2, pos_def());
    auto model = (Sigma |= wishart(V, 2.),
                  y |= normal(0., Sigma));
    using program_t = util::convert_to_program_t<std::decay_t<decltype(model)>>;
    program_t program = model;
    auto res = program.activate();

    vals.resize(std::get<0>(res).uc_offset);
    adjs.resize(std::get<0>(res).uc_offset);
    Eigen::VectorXd constrained(std::get<0>(res).c_offset);
    Eigen::Matrix<size_t, Eigen::Dynamic, 1> visit(std::get<0>(res).v_offset);

    vals << std::log(2.), 1., std::log(2.);
    adjs.setZero();
    constrained.setZero();
    visit.setZero();

    ptr_pack.uc_val = vals.data();
    ptr_pack.uc_adj = adjs.data();
    ptr_pack.c_val = constrained.data();
    ptr_pack.v_val = visit.data();
    auto expr = ad::bind(program.ad_log_pdf(ptr_pack));

    EXPECT_EQ(vals.size(), 3);
    EXPECT_EQ(adjs.size(), 3);
    EXPECT_EQ(constrained.size(), 8);
    EXPECT_EQ(visit.size(), 1);

    value_t value = ad::autodiff(expr);

    EXPECT_EQ(visit[0], 0ul);

    double logj = 0.;
    for (size_t k = 0; k < Sigma.rows(); ++k) {
        size_t idx = (Sigma.rows() * k) - (k * (k-1)) / 2;
        logj += (Sigma.rows() - k + 1) * vals(idx);
    }

    Eigen::Map<Eigen::MatrixXd> s_view(constrained.data() + Sigma.size(),
                                       Sigma.rows(), Sigma.cols());

    EXPECT_DOUBLE_EQ(value, 
            -4.818239983121333 + logj +
            math::normal_log_pdf(y.get(), 0., s_view) + 
            2*math::LOG_SQRT_TWO_PI);
}

TEST_F(ad_integration_fixture, ad_log_pdf_tp_model)
{
    scl_tp_t theta_prime;
    auto tp = (
        theta_prime = 1.,
        theta_prime += theta
    );
    auto model = (
        theta |= normal(-1., 0.5),
        x |= normal(theta_prime, 1.)
    );
    auto program = tp | model;
    auto pack = program.activate();

    EXPECT_EQ(std::get<0>(pack).uc_offset, 1ul);
    EXPECT_EQ(std::get<0>(pack).c_offset, 0ul);
    EXPECT_EQ(std::get<0>(pack).tp_offset, 1ul);
    EXPECT_EQ(std::get<0>(pack).v_offset, 0ul);

    EXPECT_EQ(std::get<1>(pack).uc_offset, 0ul);
    EXPECT_EQ(std::get<1>(pack).c_offset, 0ul);
    EXPECT_EQ(std::get<1>(pack).tp_offset, 0ul);
    EXPECT_EQ(std::get<1>(pack).v_offset, 0ul);

    Eigen::VectorXd tp_vals(1);
    Eigen::VectorXd tp_adjs(1);
    tp_adjs.setZero();
    ptr_pack.tp_val = tp_vals.data();
    ptr_pack.tp_adj = tp_adjs.data();

    vals[0] = 0.42;

    auto expr = ad::bind(program.ad_log_pdf(ptr_pack));
    double value = ad::autodiff(expr);

    double actual = -4.8442528194400545;
    EXPECT_DOUBLE_EQ(value, actual);
    
    double actual_adj = -7 * (vals[0] + 1) + x.get().sum();
    EXPECT_DOUBLE_EQ(adjs[0], actual_adj);
}

TEST_F(ad_integration_fixture, stochastic_volatility)
{
    Data<value_t, vec> y(2);
    Param phi = make_param<value_t>(bounded(-1., 1.));
    Param sigma = make_param<value_t>(lower(0.));
    Param<value_t> mu;
    Param<value_t, vec> h_std(2);
    TParam<value_t, vec> h(2);

    auto tp_expr = (
        h = h_std * sigma,
        h[0] /= ppl::sqrt(1. - phi * phi),
        h += mu,
        ppl::for_each(ppl::util::counting_iterator<>(1),
                      ppl::util::counting_iterator<>(h.size()),
                      [&](size_t i) { return h[i] += phi * (h[i-1] - mu); })
    );

    auto model = (
        phi |= ppl::uniform(-1., 1.),
        sigma |= ppl::cauchy(0., 5.),
        mu |= ppl::cauchy(0., 10.),
        h_std |= ppl::normal(0., 1.),
        y |= ppl::normal(0., ppl::exp(h / 2.))
    );

    auto program = tp_expr | model;
    auto pack = program.activate();

    EXPECT_EQ(std::get<0>(pack).uc_offset, 5ul);
    EXPECT_EQ(std::get<0>(pack).c_offset, 2ul);
    EXPECT_EQ(std::get<0>(pack).tp_offset, 2ul);
    EXPECT_EQ(std::get<0>(pack).v_offset, 2ul);

    EXPECT_EQ(std::get<1>(pack).uc_offset, 0ul);
    EXPECT_EQ(std::get<1>(pack).c_offset, 0ul);
    EXPECT_EQ(std::get<1>(pack).tp_offset, 0ul);
    EXPECT_EQ(std::get<1>(pack).v_offset, 0ul);

    y.get() << 0.4, 0.5;

    vals.resize(5);
    adjs.resize(5);
    Eigen::VectorXd tp_vals(2);
    Eigen::VectorXd tp_adjs(2);
    Eigen::VectorXd c_vals(2);
    Eigen::Matrix<size_t, Eigen::Dynamic, 1> v_vals(2);

    adjs.setZero();
    tp_adjs.setZero();
    v_vals.setZero();

    auto cb = bounded(-1.,1.);
    auto lb = lower(0.);
    cb.transform(0.95, vals[0]);    // phi
    lb.transform(0.25, vals[1]);    // sigma
    vals[2] = -1.02;                // mu
    vals[3] = 1.;                   // h_std
    vals[4] = -1.;

    ptr_pack.uc_val = vals.data();
    ptr_pack.uc_adj = adjs.data();
    ptr_pack.c_val = c_vals.data();
    ptr_pack.tp_val = tp_vals.data();
    ptr_pack.tp_adj = tp_adjs.data();
    ptr_pack.v_val = v_vals.data();

    auto expr = ad::bind(program.ad_log_pdf(ptr_pack));

    value_t res = ad::evaluate(expr);

    EXPECT_TRUE((v_vals.array() == 0ul).all());
    EXPECT_NEAR(tp_vals[0], -0.219359230974564556, tol);
    EXPECT_NEAR(tp_vals[1], -0.509391269425836346, tol);
    EXPECT_DOUBLE_EQ(res, -9.968643516356458179);

    res = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(res, -9.968643516356458179);
    EXPECT_NEAR(adjs[0], -1.219145045158940288, tol);   // phi
    EXPECT_NEAR(adjs[1], 0.525373726589360102, tol);    // sigma
    EXPECT_NEAR(adjs[2], -0.672153049338236386, tol);   // mu
    EXPECT_NEAR(adjs[3], -1.542630061308654543, tol);   // h_std[0]
    EXPECT_NEAR(adjs[4], 0.927008680929915396, tol);    // h_std[1]

    adjs.setZero();
    tp_adjs.setZero();
    res = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(res, -9.968643516356458179);
    EXPECT_NEAR(adjs[0], -1.219145045158940288, tol);   // phi
    EXPECT_NEAR(adjs[1], 0.525373726589360102, tol);    // sigma
    EXPECT_NEAR(adjs[2], -0.672153049338236386, tol);   // mu
    EXPECT_NEAR(adjs[3], -1.542630061308654543, tol);   // h_std[0]
    EXPECT_NEAR(adjs[4], 0.927008680929915396, tol);    // h_std[1]
}

} // namespace ppl
