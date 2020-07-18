#include "gtest/gtest.h"
#include <autoppl/expression/variable/constant.hpp>
#include <autoppl/util/traits/var_expr_traits.hpp>
#include <autoppl/util/traits/mock_types.hpp>

namespace ppl {
namespace expr {

struct constant_fixture : ::testing::Test
{
protected:
    static constexpr double defval = 0.3;
    using value_t = double;
    value_t c = defval;
    Constant<value_t> x{c};
};

TEST_F(constant_fixture, ctor)
{
    static_assert(util::is_var_expr_v<Constant<value_t>>);
}

TEST_F(constant_fixture, value)
{
    // first parameter ignored and was chosen arbitrarily
    EXPECT_DOUBLE_EQ(x.value(0), defval);
    c = 3.41;
    EXPECT_DOUBLE_EQ(x.value(0), defval);
}

TEST_F(constant_fixture, size) 
{
    EXPECT_EQ(x.size(), 1ul);
}

TEST_F(constant_fixture, to_ad)
{
    // Note: arbitrarily first 2 inputs (will ignore)
    auto expr = x.to_ad(0,0,0);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr), defval);
}

} // namespace expr
} // namespace ppl
