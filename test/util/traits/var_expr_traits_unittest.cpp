#include "gtest/gtest.h"
#include <autoppl/util/traits/var_expr_traits.hpp>
#include <autoppl/util/traits/mock_types.hpp>

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
    static_assert(!is_var_expr_v<MockNotVarExpr>);
    static_assert(is_shape_v<MockNotVarExpr>);
    static_assert(!var_expr_is_base_of_v<MockNotVarExpr>);
    static_assert(!has_type_value_t_v<MockNotVarExpr>);
    //static_assert(!has_func_value_v<MockNotVarExpr>);
}

} // namespace util
} // namespace ppl
