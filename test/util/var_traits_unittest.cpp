#include "gtest/gtest.h"
#include <autoppl/util/var_traits.hpp>
#include <testutil/mock_types.hpp>

namespace ppl {
namespace util {

struct var_traits_fixture : ::testing::Test
{
protected:
};

TEST_F(var_traits_fixture, is_var_v_true)
{
    static_assert(assert_is_var_v<MockParam>);
}

TEST_F(var_traits_fixture, is_var_v_false)
{
    static_assert(!is_var_v<MockParam_no_convertible>);
}

} // namespace util
} // namespace ppl
