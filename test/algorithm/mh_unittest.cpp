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
    static constexpr size_t sample_size = 20000;
    std::array<double, sample_size> storage = {0.};
    Variable<double> theta, x;
    size_t burn = 1000;

    mh_fixture()
        : theta{storage.data()}
    {}

    double sample_average()
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
    mh_posterior(model, sample_size, 1.0, 0.);
    plot_hist(storage);
    EXPECT_NEAR(sample_average(), 0., 0.1);
}

TEST_F(mh_fixture, sample_uniform)
{
    auto model = (theta |= uniform(0., 1.));
    mh_posterior(model, sample_size, 1.0, 0.);
    plot_hist(storage, 0.05, 0., 1.);
    EXPECT_NEAR(sample_average(), 0.5, 0.1);
}

TEST_F(mh_fixture, sample_unif_normal_posterior)
{
    x.observe(3.);
    auto model = (
        theta |= uniform(-20., 20.),
        x |= normal(theta, 1.)
    );
    mh_posterior(model, sample_size, 1.0, 0.);
    plot_hist(storage);
    EXPECT_NEAR(sample_average(), 3.0, 0.1);
}

} // namespace ppl
