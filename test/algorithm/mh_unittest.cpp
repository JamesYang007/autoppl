#include "gtest/gtest.h"
#include <array>
#include <limits>
#include <autoppl/algorithm/mh.hpp>
#include <autoppl/expr_builder.hpp>
#include <testutil/sample_tools.hpp>

namespace ppl {

TEST(mh_unittest, plot_hist_sanity)
{
    static constexpr size_t sample_size = 20000;
    std::array<double, sample_size> storage = {0.};
    std::normal_distribution normal_sampler(0., 1.);
    std::mt19937 gen;
    for (size_t i = 0; i < sample_size; ++i) {
        storage[i] = normal_sampler(gen);
    }
    plot_hist(storage);
}

/*
 * Fixture for Metropolis-Hastings 
 */
struct mh_fixture : ::testing::Test
{
protected:
    static constexpr size_t sample_size = 20000;
    std::array<double, sample_size> storage = {0.};
    Variable<double> theta, x;

    mh_fixture()
        : theta{storage.data()}
    {}

    double sample_average(size_t burn)
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
    mh_posterior(model, sample_size);
    plot_hist(storage);
    EXPECT_NEAR(sample_average(1000), 0., 0.01);
}

TEST_F(mh_fixture, sample_uniform)
{
    auto model = (theta |= uniform(0., 1.));
    mh_posterior(model, sample_size);
    plot_hist(storage, 0.05, 0., 1.);
    EXPECT_NEAR(sample_average(1000), 0.5, 0.01);
}

TEST_F(mh_fixture, sample_unif_normal_posterior)
{
    x.observe(3.);
    auto model = (
        theta |= uniform(-20., 20.),
        x |= normal(theta, 1.)
    );
    mh_posterior(model, sample_size);
    plot_hist(storage);
    EXPECT_NEAR(sample_average(1000), 3.0, 0.01);
}

} // namespace ppl
