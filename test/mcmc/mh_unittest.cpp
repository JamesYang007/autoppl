#include "gtest/gtest.h"
#include <array>
#include <limits>
#include <testutil/base_fixture.hpp>
#include <testutil/sample_tools.hpp>
#include <autoppl/mcmc/mh/mh.hpp>
#include <autoppl/expression/program/program.hpp>
#include <autoppl/expression/model/bar_eq.hpp>
#include <autoppl/expression/model/glue.hpp>
#include <autoppl/expression/variable/data.hpp>
#include <autoppl/expression/variable/param.hpp>
#include <autoppl/expression/variable/binary.hpp>
#include <autoppl/expression/variable/constant.hpp>
#include <autoppl/expression/distribution/bernoulli.hpp>
#include <autoppl/expression/distribution/uniform.hpp>
#include <autoppl/expression/distribution/normal.hpp>
#include <autoppl/expression/op_overloads.hpp>

namespace ppl {

/*
 * Fixture for Metropolis-Hastings 
 */
struct mh_fixture: 
    base_fixture<util::cont_param_t>,
    base_fixture<util::disc_param_t>,
    ::testing::Test
{
protected:
    using cont_base_t = base_fixture<util::cont_param_t>;
    using disc_base_t = base_fixture<util::disc_param_t>;
    using cont_value_t = typename cont_base_t::value_t;
    using disc_value_t = typename disc_base_t::value_t;

    using p_cont_scl_t = Param<cont_value_t>;
    using p_cont_vec_t = Param<cont_value_t, ppl::vec>;
    using p_disc_scl_t = Param<disc_value_t>;

    using d_cont_scl_t = Data<cont_value_t>;
    using d_cont_vec_t = Data<cont_value_t, ppl::vec>;
    using d_disc_scl_t = Data<disc_value_t>;

    size_t sample_size = 20000;
    size_t warmup = 1000;

    MHConfig config;

    p_cont_scl_t theta, theta_2;
    d_cont_vec_t y;

    mh_fixture()
        : theta{}
        , theta_2{}
        , y(5)
    {
        y.get() << 0.1, 0.2, 0.3, 0.4, 0.5;

        config.warmup = warmup;
        config.samples = sample_size;
        config.seed = 0;
    }

    template <class ArrayType>
    double sample_average(const ArrayType& storage)
    {
        double sum = std::accumulate(
                storage.data(),  
                storage.data() + storage.size(), 
                0.);
        return sum / storage.size();
    }
};

TEST_F(mh_fixture, sample_std_normal)
{
    auto model = (theta |= normal(0., 1.));
    auto out = mh(model, config);
    plot_hist(out.cont_samples.col(0));
    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 0., 0.1);
}

TEST_F(mh_fixture, sample_uniform)
{
    auto model = (theta |= uniform(0., 1.));
    auto out = mh(model, config);
    plot_hist(out.cont_samples.col(0), 0.1, 0., 1.);
    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 0.5, 0.1);
}

TEST_F(mh_fixture, sample_bern)
{
    p_disc_scl_t theta;
    auto model = (
        theta |= bernoulli(0.02)
    );
    auto out = mh(model, config);
    plot_hist(out.disc_samples.col(0), 0.2, 0., 1.);
    EXPECT_NEAR(out.disc_samples.col(0).cast<double>().mean(), 0.02, 0.01);
}

TEST_F(mh_fixture, sample_unif_normal_posterior_mean)
{
    d_cont_scl_t x(3.);
    auto model = (
        theta |= uniform(-20., 20.),
        x |= normal(theta, 1.)
    );
    auto out = mh(model, config);
    plot_hist(out.cont_samples.col(0));
    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 3.0, 0.1);
}

TEST_F(mh_fixture, sample_unif_normal_posterior_stddev)
{
    d_cont_scl_t x(3.14);
    auto model = (
        theta |= uniform(0.1, 5.),
        x |= normal(0., theta)
    );
    auto out = mh(model, config);
    plot_hist(out.cont_samples.col(0), 0.2);
    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 3.27226, 0.1);
}

TEST_F(mh_fixture, sample_unif_normal_posterior_mean_stddev)
{
    d_cont_scl_t x(-0.314);
    auto model = (
        theta |= normal(0., 1.),
        theta_2 |= uniform(0.1, 5.),
        x |= normal(theta, theta_2)
    );
    auto out = mh(model, config);
    plot_hist(out.cont_samples.col(0));
    plot_hist(out.cont_samples.col(1), 0.2);
    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), -0.1235305689822228, 0.1);
    EXPECT_NEAR(sample_average(out.cont_samples.col(1)), 1.868814361437099766, 0.1);
}

TEST_F(mh_fixture, sample_unif_normal_posterior_mean_samples) {
    auto model = (
        theta |= uniform(-1., 2.), 
        y |= normal(theta, 1.0) // {0.1, 0.2, 0.3, 0.4, 0.5}
    );

    auto out = mh(model, config);
    plot_hist(out.cont_samples.col(0));
    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 0.3, 0.1);
}

TEST_F(mh_fixture, sample_unif_normal_posterior_mean_std_samples) {
    auto model = (
        theta |= uniform(-1., 1.),
        theta_2 |= uniform(0., 1.),
        y |= normal(theta, theta_2) // {0.1, 0.2, 0.3, 0.4, 0.5}
    );

    auto out = mh(model, config);

    plot_hist(out.cont_samples.col(0), 0.5);
    plot_hist(out.cont_samples.col(1), 0.5);

    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 0.29951, 0.05); // found numerical with Mathematica
    EXPECT_NEAR(sample_average(out.cont_samples.col(1)), 0.241658, 0.05);
}

TEST_F(mh_fixture, sample_unif_bern_posterior_observe_zero)
{
    d_disc_scl_t x_discrete(0);
    auto model = (
        theta |= uniform(0., 1.),
        x_discrete |= bernoulli(theta)
    );
    auto out = mh(model, config);
    plot_hist(out.cont_samples.col(0), 0.2, 0., 1.);
    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 1./3., 0.1);
}

TEST_F(mh_fixture, sample_unif_bern_posterior_observe_one)
{
    d_disc_scl_t x_discrete(1);
    auto model = (
        theta |= uniform(0., 1.),
        x_discrete |= bernoulli(theta)
    );
    auto out = mh(model, config);
    plot_hist(out.cont_samples.col(0), 0.2, 0., 1.);
    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 2./3., 0.1);
}

// COMPILER ERROR: good :) discrete param should not be a continuous parameter
//TEST_F(mh_fixture, sample_bern_normal_posterior)
//{
//    p_disc_scl_t theta(disc_storage.data());
//    d_cont_scl_t x(1.);
//    auto model = (
//        theta |= bernoulli(0.5),
//        x |= normal(theta, 1.)
//    );
//    mh(model, sample_size, warmup, 1.0, 1./3, 0.);
//    plot_hist(disc_storage, 0.2, 0., 1.);
//    EXPECT_NEAR(sample_average(disc_storage), 
//                0.62245933120185456463890056, 0.1);
//}

} // namespace ppl
