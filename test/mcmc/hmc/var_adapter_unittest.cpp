#include <autoppl/mcmc/hmc/var_adapter.hpp>
#include <gtest/gtest.h>

namespace ppl {
namespace mcmc {

struct var_adapter_fixture : ::testing::Test
{
protected:
};

TEST_F(var_adapter_fixture, diag)
{
    VarAdapter<diag_var> adapter1(3, 3, 1, 1, 1);
    VarAdapter<diag_var> adapter2(3, 30, 10, 20, 10);
}

} // namespace mcmc
} // namespace ppl
