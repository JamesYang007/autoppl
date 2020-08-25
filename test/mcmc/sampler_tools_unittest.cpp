#include "gtest/gtest.h"
#include <array>
#include <random>
#include <autoppl/expression/model/bar_eq.hpp>
#include <autoppl/expression/model/glue.hpp>
#include <autoppl/expression/variable/data.hpp>
#include <autoppl/expression/variable/param.hpp>
#include <autoppl/expression/variable/binary.hpp>
#include <autoppl/expression/variable/constant.hpp>
#include <autoppl/expression/distribution/bernoulli.hpp>
#include <autoppl/expression/distribution/uniform.hpp>
#include <autoppl/expression/distribution/normal.hpp>
#include <autoppl/expression/program/program.hpp>
#include <autoppl/expression/op_overloads.hpp>
#include <autoppl/mcmc/sampler_tools.hpp>

namespace ppl {
namespace mcmc {

struct sampler_tools_fixture : ::testing::Test
{
protected:
    using cont_value_t = util::cont_param_t;
    using disc_value_t = util::disc_param_t;
    using cont_param_t = Param<cont_value_t, ppl::vec>;
    using disc_param_t = Param<disc_value_t, ppl::vec>;

    static constexpr size_t size = 3;

    std::array<disc_value_t, size> disc_values;
    std::array<cont_value_t, size> cont_values;
    cont_param_t cw = size;
    disc_param_t dw = size;

    std::mt19937 gen;

    sampler_tools_fixture()
        : disc_values{{0, 1, 1}}
        , cont_values{{-3., 0.2, 13.23}}
    {}
};

TEST_F(sampler_tools_fixture, init_param_disc)
{
    auto model = (dw |= bernoulli(0.5));
    using program_t = util::convert_to_program_t<std::decay_t<decltype(model)>>;
    program_t program = model;
    program.activate();
    program.bind(util::make_ptr_pack(disc_values.data()));
    program.init_params(gen);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(disc_values[i], 0);
    }
}

TEST_F(sampler_tools_fixture, init_param_cont_unbounded)
{
    auto model = (cw |= normal(0., 1.));
    using program_t = util::convert_to_program_t<std::decay_t<decltype(model)>>;
    program_t program = model;
    program.activate();
    program.bind(util::make_ptr_pack(cont_values.data()));
    program.init_params(gen);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_LT(-2., cont_values[i]);
        EXPECT_LT(cont_values[i], 2.);
    }
}

//TEST_F(sampler_tools_fixture, init_param_cont_bounded)
//{
//    cont_value_t min = 0.;
//    cont_value_t max = 0.000001;
//    auto model = (cw |= uniform(min, max));
//    expr::activate(model);
//    model.bind(cont_values.data(), nullptr, nullptr);
//    init_params(model, gen);
//    for (size_t i = 0; i < size; ++i) {
//        EXPECT_LE(min, cont_values[i]);
//        EXPECT_LE(cont_values[i], max);
//    }
//}

} // namespace mcmc
} // namespace ppl
