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
#if __cplusplus <= 201703L
    static_assert(assert_is_var_expr_v<MockVarExpr>);
#else
    static_assert(var_expr<MockVarExpr>);
#endif
}

TEST_F(var_expr_traits_fixture, is_var_expr_v_false)
{
#if __cplusplus <= 201703L
    static_assert(!is_var_expr_v<MockVarExpr_no_convertible>);
#else
    static_assert(!var_expr<MockVarExpr_no_convertible>);
#endif
}

} // namespace util
} // namespace ppl
