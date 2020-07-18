#include "gtest/gtest.h"
#include <array>
#include <limits>
#include <autoppl/mcmc/mh.hpp>
#include <autoppl/expression/expr_builder.hpp>
#include <testutil/sample_tools.hpp>

namespace ppl {

/*
 * Fixture for Metropolis-Hastings 
 */
struct mh_fixture : ::testing::Test
{
protected:
    using cont_value_t = double;
    using disc_value_t = int;
    using p_cont_scl_t = Param<cont_value_t>;
    using p_cont_vec_t = Param<cont_value_t, ppl::vec>;
    using p_disc_scl_t = Param<disc_value_t>;
    using d_cont_scl_t = Data<cont_value_t>;
    using d_disc_scl_t = Data<disc_value_t>;
    using d_cont_vec_t = Data<cont_value_t, ppl::vec>;

    size_t sample_size = 20000;
    size_t warmup = 1000;
    std::vector<cont_value_t> cont_storage, cont_storage_2;
    std::vector<disc_value_t> disc_storage, disc_storage_2;
    p_cont_scl_t theta, theta_2;
    d_cont_vec_t y {0.1, 0.2, 0.3, 0.4, 0.5};

    mh_fixture()
        : cont_storage(sample_size)
        , cont_storage_2(sample_size)
        , disc_storage(sample_size)
        , disc_storage_2(sample_size)
        , theta{cont_storage.data()}
        , theta_2{cont_storage_2.data()}
    {}

    template <class ArrayType>
    double sample_average(const ArrayType& storage)
    {
        double sum = std::accumulate(
                std::next(storage.begin(), warmup), 
                storage.end(), 
                0.);
        return sum / (storage.size() - warmup);
    }
};

TEST_F(mh_fixture, sample_std_normal)
{
    auto model = (theta |= normal(0., 1.));
    mh(model, sample_size, warmup, 1.0, 0.25, 0);
    plot_hist(cont_storage);
    EXPECT_NEAR(sample_average(cont_storage), 0., 0.1);
}

TEST_F(mh_fixture, sample_uniform)
{
    auto model = (theta |= uniform(0., 1.));
    mh(model, sample_size, warmup, 1.0, 0.25, 0);
    plot_hist(cont_storage, 0.1, 0., 1.);
    EXPECT_NEAR(sample_average(cont_storage), 0.5, 0.1);
}

TEST_F(mh_fixture, sample_unif_normal_posterior_mean)
{
    d_cont_scl_t x(3.);
    auto model = (
        theta |= uniform(-20., 20.),
        x |= normal(theta, 1.)
    );
    mh(model, sample_size, warmup, 1.0, 0.25, 0.);
    plot_hist(cont_storage);
    EXPECT_NEAR(sample_average(cont_storage), 3.0, 0.1);
}

TEST_F(mh_fixture, sample_unif_normal_posterior_stddev)
{
    d_cont_scl_t x(3.14);
    auto model = (
        theta |= uniform(0.1, 5.),
        x |= normal(0., theta)
    );
    mh(model, sample_size, warmup, 0.5, 0.25, 0.);
    plot_hist(cont_storage, 0.2);
    EXPECT_NEAR(sample_average(cont_storage), 3.27226, 0.1);
}

TEST_F(mh_fixture, sample_unif_normal_posterior_mean_stddev)
{
    d_cont_scl_t x(-0.314);
    auto model = (
        theta |= normal(0., 1.),
        theta_2 |= uniform(0.1, 5.),
        x |= normal(theta, theta_2)
    );
    mh(model, sample_size, warmup, 0.5, 0.25, 0.);
    plot_hist(cont_storage);
    plot_hist(cont_storage_2, 0.2);
    EXPECT_NEAR(sample_average(cont_storage), -0.1235305689822228, 0.1);
    EXPECT_NEAR(sample_average(cont_storage_2), 1.868814361437099766, 0.1);
}

TEST_F(mh_fixture, sample_unif_normal_posterior_mean_samples) {
    auto model = (
        theta |= uniform(-1., 2.), 
        y |= normal(theta, 1.0) // {0.1, 0.2, 0.3, 0.4, 0.5}
    );

    mh(model, sample_size, warmup, 0.5, 0.25, 0.);
    plot_hist(cont_storage);
    EXPECT_NEAR(sample_average(cont_storage), 0.3, 0.1);
}

TEST_F(mh_fixture, sample_unif_normal_posterior_mean_std_samples) {
    auto model = (
        theta |= uniform(-1., 1.),
        theta_2 |= uniform(0., 1.),
        y |= normal(theta, theta_2) // {0.1, 0.2, 0.3, 0.4, 0.5}
    );

    mh(model, sample_size, warmup, 0.5, 0.25, 0.);

    plot_hist(cont_storage, 0.5);
    plot_hist(cont_storage_2, 0.5);

    EXPECT_NEAR(sample_average(cont_storage), 0.29951, 0.05); // found numerical with Mathematica
    EXPECT_NEAR(sample_average(cont_storage_2), 0.241658, 0.05);
}

TEST_F(mh_fixture, sample_unif_bern_posterior_observe_zero)
{
    d_disc_scl_t x_discrete(0);
    auto model = (
        theta |= uniform(0., 1.),
        x_discrete |= bernoulli(theta)
    );
    mh(model, sample_size, warmup, 1.0, 0.25, 0.);
    plot_hist(cont_storage, 0.2, 0., 1.);
    EXPECT_NEAR(sample_average(cont_storage), 1./3., 0.1);
}

TEST_F(mh_fixture, sample_unif_bern_posterior_observe_one)
{
    d_disc_scl_t x_discrete(1);
    auto model = (
        theta |= uniform(0., 1.),
        x_discrete |= bernoulli(theta)
    );
    mh(model, sample_size, warmup, 1.0, 0.25, 0.);
    plot_hist(cont_storage, 0.2, 0., 1.);
    EXPECT_NEAR(sample_average(cont_storage), 2./3., 0.1);
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
