#include "gtest/gtest.h"
#include <autoppl/expression/variable/constant.hpp>
#include <autoppl/util/var_expr_traits.hpp>
#include <testutil/mock_types.hpp>

namespace ppl {
namespace expr {

struct constant_fixture : ::testing::Test
{
protected:
    using value_t = double;
    value_t c = 0.3;
    Constant<value_t> x{c};
};

TEST_F(constant_fixture, ctor)
{
    static_assert(util::assert_is_var_expr_v<Constant<value_t>>);
}

TEST_F(constant_fixture, convertible_value)
{
    EXPECT_EQ(x.get_value(0), 0.3);
    c = 3.41;
    EXPECT_EQ(x.get_value(0), 0.3);
}

} // namespace expr
} // namespace ppl
