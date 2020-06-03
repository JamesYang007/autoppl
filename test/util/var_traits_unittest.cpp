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
#if __cplusplus <= 201703L
    static_assert(assert_is_var_v<MockParam>);
#else
    static_assert(param<MockParam>);
    static_assert(var<MockParam>);
#endif
}

TEST_F(var_traits_fixture, is_var_v_false)
{
#if __cplusplus <= 201703L
    static_assert(!is_var_v<MockParam_no_convertible>);
#else
    static_assert(!param<MockParam_no_convertible>);
    static_assert(!var<MockParam_no_convertible>);
#endif
}

} // namespace util
} // namespace ppl
