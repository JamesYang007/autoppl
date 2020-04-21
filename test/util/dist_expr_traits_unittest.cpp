#include "gtest/gtest.h"
#include <autoppl/util/dist_expr_traits.hpp>
#include <testutil/mock_types.hpp>

namespace ppl {
namespace util {

struct dist_expr_traits_fixture : ::testing::Test
{
protected:
};

TEST_F(dist_expr_traits_fixture, is_dist_expr_v_true)
{
    static_assert(assert_is_dist_expr_v<MockDistExpr>);
}

TEST_F(dist_expr_traits_fixture, is_dist_expr_v_false)
{
    static_assert(!is_dist_expr_v<MockDistExpr_no_pdf>);
    static_assert(!is_dist_expr_v<MockDistExpr_no_log_pdf>);
}

} // namespace util
} // namespace ppl
