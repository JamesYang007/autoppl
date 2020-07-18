#include "gtest/gtest.h"
#include <array>
#include <autoppl/math/math.hpp>

namespace ppl {
namespace math {

struct math_fixture : ::testing::Test
{
protected:
    std::array<double, 3> x = {0};
};

TEST_F(math_fixture, min_edge_case)
{
    auto res = min(x.end(), x.begin());
    EXPECT_DOUBLE_EQ(res, inf<double>);
}

TEST_F(math_fixture, max_edge_case)
{
    auto res = max(x.end(), x.begin());
    EXPECT_DOUBLE_EQ(res, neg_inf<double>);
}

} // namespace math
} // namespace ppl
