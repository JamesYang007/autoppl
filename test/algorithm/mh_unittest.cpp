#include "gtest/gtest.h"
#include <chrono>
#include <array>
#include <autoppl/algorithm/mh.hpp>
#include <autoppl/expr_builder.hpp>

namespace ppl {

template <class ArrayType>
inline void plot_hist(const ArrayType& arr,
                      double step_size = .5)
{
    constexpr size_t nstars = 100;    // maximum number of stars to distribute

    const int64_t min = std::floor(*std::min_element(arr.begin(), arr.end()));
    const int64_t max = std::ceil(*std::max_element(arr.begin(), arr.end()));
    const uint64_t range = max - min;
    const uint64_t n_hist = std::floor(range/step_size);

    // keeps count for each histogram bar
    std::vector<uint64_t> counter(n_hist, 0);

    for (auto x : arr) {
        ++counter[std::floor((x - min) / step_size)];
    }

    EXPECT_EQ(std::accumulate(counter.begin(), counter.end(), 0), (int) arr.size());

    for (size_t i = 0; i < n_hist; ++i) {
        std::cout << i << "-" << (i+1) << ": " << '\t';
        std::cout << std::string(counter[i] * nstars/arr.size(), '*') << std::endl;
    }

}

TEST(mh_unittest, plot_hist_sanity)
{
    static constexpr size_t sample_size = 5000;
    std::array<double, sample_size> storage = {0.};
    std::normal_distribution normal_sampler(0., 1.);
    std::mt19937 gen;
    for (size_t i = 0; i < sample_size; ++i) {
        storage[i] = normal_sampler(gen);
    }
    plot_hist(storage);
}

struct mh_fixture : ::testing::Test
{
protected:
    static constexpr size_t sample_size = 5000;
    std::array<double, sample_size> storage = {0.};
    Variable<double> theta;

    mh_fixture()
        : theta{storage.data()}
    {}
};

TEST_F(mh_fixture, sample_std_normal)
{
    auto model = (theta |= normal(0., 1.));
    mh_posterior(model, sample_size);
    plot_hist(storage);
}

TEST_F(mh_fixture, sample_uniform)
{
    auto model = (theta |= uniform(0., 1.));
    mh_posterior(model, sample_size);
    plot_hist(storage, 0.05);
}

} // namespace ppl
