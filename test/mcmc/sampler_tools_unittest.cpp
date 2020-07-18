#include "gtest/gtest.h"
#include <array>
#include <random>
#include <autoppl/expression/expr_builder.hpp>
#include <autoppl/expression/activate.hpp>
#include <autoppl/mcmc/sampler_tools.hpp>

namespace ppl {
namespace mcmc {

struct sampler_tools_fixture : ::testing::Test
{
protected:
    using cont_value_t = double;
    using disc_value_t = int;
    using cont_param_t = Param<cont_value_t, ppl::vec>;
    using disc_param_t = Param<disc_value_t, ppl::vec>;

    static constexpr size_t size = 3;

    std::array<disc_value_t, size> disc_values;
    std::array<cont_value_t, size> cont_values;
    std::array<cont_value_t, size> cont_one_samples;
    cont_param_t cw = size;
    disc_param_t dw = size;

    std::mt19937 gen;

    sampler_tools_fixture()
        : disc_values{{0, 1, 1}}
        , cont_values{{-3., 0.2, 13.23}}
        , cont_one_samples{{0,0,0}}
    {
        for (size_t i = 0; i < size; ++i) {
            cw.storage(i) = &cont_one_samples[i];
        }
    }
};

TEST_F(sampler_tools_fixture, init_param_disc)
{
    auto model = (dw |= bernoulli(0.5));
    expr::activate(model);
    init_params(model, gen, disc_values);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_LE(0, disc_values[i]);
        EXPECT_LE(disc_values[i], 1);
    }
}

TEST_F(sampler_tools_fixture, init_param_cont_unbounded)
{
    auto model = (cw |= normal(0., 1.));
    expr::activate(model);
    init_params(model, gen, cont_values);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_LT(math::neg_inf<cont_value_t>, cont_values[i]);
        EXPECT_LT(cont_values[i], math::inf<cont_value_t>);
    }
}

TEST_F(sampler_tools_fixture, init_param_cont_bounded)
{
    cont_value_t min = 0.;
    cont_value_t max = 0.000001;
    auto model = (cw |= uniform(min, max));
    expr::activate(model);
    init_params(model, gen, cont_values);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_LE(min, cont_values[i]);
        EXPECT_LE(cont_values[i], max);
    }
}

TEST_F(sampler_tools_fixture, store_sample)
{
    auto model = (cw |= normal(0., 1.));
    expr::activate(model);
    store_sample(model, cont_values, 0); // store first sample
    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(cont_one_samples[i], cont_values[i]);
    }
}

} // namespace mcmc
} // namespace ppl
