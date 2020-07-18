#include "gtest/gtest.h"
#include <autoppl/util/traits/var_traits.hpp>
#include <autoppl/util/traits/mock_types.hpp>

namespace ppl {
namespace util {

struct var_traits_fixture : ::testing::Test
{
protected:
};

TEST_F(var_traits_fixture, is_var_v_true)
{
    static_assert(is_var_v<MockParam>);
    static_assert(is_param_v<MockParam>);
    static_assert(is_var_v<MockData>);
    static_assert(is_data_v<MockData>);
}

TEST_F(var_traits_fixture, is_var_v_false)
{
    static_assert(!is_param_v<MockNotParam>);
    static_assert(!is_var_v<MockNotParam>);
    static_assert(is_var_expr_v<MockNotParam>);
    static_assert(!param_is_base_of_v<MockNotParam>);
    static_assert(!has_type_id_t_v<MockNotParam>);
    static_assert(!has_type_pointer_t_v<MockNotParam>);
    static_assert(!has_type_const_pointer_t_v<MockNotParam>);
    static_assert(!has_func_id_v<MockNotParam>);

    static_assert(!is_data_v<MockNotData>);
    static_assert(!is_var_v<MockNotData>);
    static_assert(is_var_expr_v<MockNotData>);
    static_assert(!data_is_base_of_v<MockNotData>);
    static_assert(!has_type_id_t_v<MockNotData>);
    static_assert(!has_func_id_v<MockNotData>);
}

} // namespace util
} // namespace ppl
