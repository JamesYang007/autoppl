#include "gtest/gtest.h"
#include <array>
#include <autoppl/expr_builder.hpp>
#include <autoppl/mcmc/hmc/momentum_handler.hpp>
#include <autoppl/mcmc/sampler_tools.hpp>

namespace ppl {
namespace mcmc {

struct sampler_tools_fixture : ::testing::Test
{
protected:
    using var_t = Param<double>;

    static constexpr size_t n_params = 10;
    std::array<Param<double>, n_params> thetas;
    Data<double> x;

    sampler_tools_fixture()
    {}
};

} // namespace mcmc
} // namespace ppl
