#include "gtest/gtest.h"
#include <autoppl/expression/variable/variable_viewer.hpp>
#include <testutil/mock_types.hpp>

namespace ppl {
namespace expr {

struct variable_viewer_fixture : ::testing::Test
{
protected:
    using value_t = typename MockVar::value_t;
    MockVar var;
    VariableViewer<MockVar> x = var;
};

TEST_F(variable_viewer_fixture, ctor)
{
    static_assert(util::is_var_expr_v<VariableViewer<MockVar>>);
}

TEST_F(variable_viewer_fixture, convertible_value)
{
    var.set_value(1.);
    EXPECT_EQ(static_cast<value_t>(x), 1.);

    // Tests if viewer correctly reflects any changes that happened in var.
    var.set_value(-3.14);
    EXPECT_EQ(static_cast<value_t>(x), -3.14);
}

} // namespace expr
} // namespace ppl
