#include "gtest/gtest.h"
#include <autoppl/util/var_expr_traits.hpp>
#include <testutil/mock_types.hpp>

namespace ppl {
namespace util {

struct var_expr_traits_fixture : ::testing::Test
{
protected:
};

TEST_F(var_expr_traits_fixture, is_var_expr_v_true)
{
    static_assert(is_var_expr_v<MockVarExpr>);
}

TEST_F(var_expr_traits_fixture, is_var_expr_v_false)
{
    static_assert(!is_var_expr_v<MockVarExpr_no_convertible>);
}

} // namespace util
} // namespace ppl
