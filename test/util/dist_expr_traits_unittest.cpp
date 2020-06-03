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
#if __cplusplus <= 201703L
    static_assert(assert_is_dist_expr_v<MockDistExpr>);
#else
    static_assert(dist_expr<MockDistExpr>);
#endif
}

TEST_F(dist_expr_traits_fixture, is_dist_expr_v_false)
{
#if __cplusplus <= 201703L
    static_assert(!is_dist_expr_v<MockDistExpr_no_pdf>);
    static_assert(!is_dist_expr_v<MockDistExpr_no_log_pdf>);
#else
    static_assert(!dist_expr<MockDistExpr_no_pdf>);
    static_assert(!dist_expr<MockDistExpr_no_log_pdf>);
#endif
}

} // namespace util
} // namespace ppl
