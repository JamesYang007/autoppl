#include "gtest/gtest.h"
#include <autoppl/util/traits/dist_expr_traits.hpp>
#include <autoppl/util/traits/mock_types.hpp>

namespace ppl {
namespace util {

struct dist_expr_traits_fixture : ::testing::Test
{
protected:
};

TEST_F(dist_expr_traits_fixture, is_dist_expr_v_true)
{
    static_assert(is_dist_expr_v<MockDistExpr>);
}

} // namespace util
} // namespace ppl
