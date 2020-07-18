#include "gtest/gtest.h"
#include <autoppl/util/traits/shape_traits.hpp>
#include <autoppl/util/traits/mock_types.hpp>

namespace ppl {
namespace util {

struct shape_traits_fixture : ::testing::Test
{
protected:
};

TEST_F(shape_traits_fixture, is_shape_v_true)
{
    static_assert(is_shape_v<MockScalar>);
    static_assert(is_scl_v<MockScalar>);
}

} // namespace util
} // namespace ppl
