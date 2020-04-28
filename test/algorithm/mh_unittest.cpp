#include "gtest/gtest.h"
#include <array>
#include <limits>
#include <autoppl/algorithm/mh.hpp>
#include <autoppl/expr_builder.hpp>
#include <testutil/sample_tools.hpp>

namespace ppl {

/*
 * Fixture for Metropolis-Hastings 
 */
struct mh_fixture : ::testing::Test
{
protected:
    size_t sample_size = 20000;
    std::vector<double> storage, storage_2;
    Param<double> theta, theta_2;
    Data<double> y {0.1, 0.2, 0.3, 0.4, 0.5};
    Data<double> x;
    Data<int> x_discrete;
    size_t burn = 1000;

    mh_fixture()
        : storage(sample_size)
        , storage_2(sample_size)
        , theta{storage.data()}
        , theta_2{storage_2.data()}
    {}

    template <class ArrayType>
    double sample_average(const ArrayType& storage)
    {
        double sum = std::accumulate(
                std::next(storage.begin(), burn), 
                storage.end(), 
                0.);
        return sum / (storage.size() - burn);
    }
};

TEST_F(mh_fixture, sample_std_normal)
{
    auto model = (theta |= normal(0., 1.));
    mh_posterior(model, sample_size, 1.0, 0.25, 0.);
    plot_hist(storage);
    EXPECT_NEAR(sample_average(storage), 0., 0.1);
    std::cout << sample_average(storage) << std::endl;
}

TEST_F(mh_fixture, sample_uniform)
{
    auto model = (theta |= uniform(0., 1.));
    mh_posterior(model, sample_size, 1.0, 0.25, 0.);
    plot_hist(storage, 0.1, 0., 1.);
    EXPECT_NEAR(sample_average(storage), 0.5, 0.1);
}

TEST_F(mh_fixture, sample_unif_normal_posterior_mean)
{
    x.observe(3.);
    auto model = (
        theta |= uniform(-20., 20.),
        x |= normal(theta, 1.)
    );
    mh_posterior(model, sample_size, 1.0, 0.25, 0.);
    plot_hist(storage);
    EXPECT_NEAR(sample_average(storage), 3.0, 0.1);
}

TEST_F(mh_fixture, sample_unif_normal_posterior_stddev)
{
    x.observe(3.14);
    auto model = (
        theta |= uniform(0.1, 5.),
        x |= normal(0., theta)
    );
    mh_posterior(model, sample_size, 0.5, 0.25, 0.);
    plot_hist(storage, 0.2);
    EXPECT_NEAR(sample_average(storage), 3.27226, 0.1);
}

TEST_F(mh_fixture, sample_unif_normal_posterior_mean_stddev)
{
    x.observe(-0.314);
    auto model = (
        theta |= normal(0., 1.),
        theta_2 |= uniform(0.1, 5.),
        x |= normal(theta, theta_2)
    );
    mh_posterior(model, sample_size, 0.5, 0.25, 0.);
    plot_hist(storage);
    plot_hist(storage_2, 0.2);
    EXPECT_NEAR(sample_average(storage), -0.1235305689822228, 0.1);
    EXPECT_NEAR(sample_average(storage_2), 1.868814361437099766, 0.1);
}

TEST_F(mh_fixture, sample_unif_normal_posterior_mean_samples) {
    auto model = (
        theta |= uniform(-1., 2.), 
        y |= normal(theta, 1.0) // {0.1, 0.2, 0.3, 0.4, 0.5}
    );

    mh_posterior(model, sample_size, 0.5, 0.25, 0.);
    plot_hist(storage);
    EXPECT_NEAR(sample_average(storage), 0.3, 0.1);
}

TEST_F(mh_fixture, sample_unif_normal_posterior_mean_std_samples) {
    auto model = (
        theta |= uniform(-1., 1.),
        theta_2 |= uniform(0., 1.),
        y |= normal(theta, theta_2) // {0.1, 0.2, 0.3, 0.4, 0.5}
    );

    mh_posterior(model, sample_size, 0.5, 0.25, 0.);

    plot_hist(storage, 0.5);
    plot_hist(storage_2, 0.5);

    EXPECT_NEAR(sample_average(storage), 0.29951, 0.05); // found numerical with Mathematica
    EXPECT_NEAR(sample_average(storage_2), 0.241658, 0.05);
}

TEST_F(mh_fixture, sample_unif_bern_posterior_observe_zero)
{
    x_discrete.observe(0);
    auto model = (
        theta |= uniform(0., 1.),
        x_discrete |= bernoulli(theta)
    );
    mh_posterior(model, sample_size, 1.0, 0.25, 0.);
    plot_hist(storage, 0.2, 0., 1.);
    EXPECT_NEAR(sample_average(storage), 1./3., 0.1);
}

TEST_F(mh_fixture, sample_unif_bern_posterior_observe_one)
{
    x_discrete.observe(1);
    auto model = (
        theta |= uniform(0., 1.),
        x_discrete |= bernoulli(theta)
    );
    mh_posterior(model, sample_size, 1.0, 0.25, 0.);
    plot_hist(storage, 0.2, 0., 1.);
    EXPECT_NEAR(sample_average(storage), 2./3., 0.1);
}

TEST_F(mh_fixture, sample_bern_normal_posterior)
{
    std::vector<int> storage(sample_size);
    Param<int> theta{storage.data()};
    x.observe(1.);
    auto model = (
        theta |= bernoulli(0.5),
        x |= normal(theta, 1.)
    );
    mh_posterior(model, sample_size, 1.0, 1./3, 0.);
    plot_hist(storage, 0.2, 0., 1.);
    EXPECT_NEAR(sample_average(storage), 0.62245933120185456463890056, 0.1);
}

} // namespace ppl
