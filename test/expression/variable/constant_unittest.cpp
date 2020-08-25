#include "gtest/gtest.h"
#include <fastad>
#include <testutil/base_fixture.hpp>
#include <autoppl/expression/variable/constant.hpp>
#include <autoppl/util/traits/var_expr_traits.hpp>

namespace ppl {
namespace expr {
namespace var {

struct constant_fixture: 
    base_fixture<double>,
    ::testing::Test
{
protected:
    value_t c = 0.3;
    Constant<value_t> x{c};
};

TEST_F(constant_fixture, ctor)
{
    static_assert(util::is_var_expr_v<Constant<value_t>>);
}

TEST_F(constant_fixture, value)
{
    value_t orig = c;
    EXPECT_DOUBLE_EQ(x.get(), orig);
    c = 3.41;
    EXPECT_DOUBLE_EQ(x.get(), orig);
}

TEST_F(constant_fixture, size) 
{
    EXPECT_EQ(x.size(), 1ul);
}

TEST_F(constant_fixture, ad)
{
    auto expr = x.ad(ptr_pack);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr), c);
}

} // namespace var
} // namespace expr
} // namespace ppl
