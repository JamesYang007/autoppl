#include "gtest/gtest.h"
#include <fastad>
#include <testutil/base_fixture.hpp>
#include <autoppl/expression/variable/data.hpp>
#include <autoppl/util/traits/var_traits.hpp>

namespace ppl {
namespace expr {
namespace var {

struct data_fixture: 
    base_fixture<double>,
    ::testing::Test
{
protected:
    static constexpr value_t defval1 = 1.0;
    static constexpr value_t defval2 = 2.0;
    static constexpr size_t size1 = 7;
    static constexpr size_t size2 = 17;

    value_t d1 = defval1;
    value_t d2 = defval2;

    std::vector<value_t> values1;
    std::vector<value_t> values2;

    data_fixture()
        : values1(size1)
        , values2(size2)
    {
        for (size_t i = 0; i < values1.size(); ++i) {
            values1[i] = i + defval1;
        }
        for (size_t i = 0; i < values2.size(); ++i) {
            values2[i] = i + defval2;
        }
    }
};

TEST_F(data_fixture, type_check)
{
    static_assert(util::is_data_v<scl_dv_t>);
    static_assert(util::is_scl_v<scl_dv_t>);

    static_assert(util::is_data_v<vec_dv_t>);
    static_assert(util::is_vec_v<vec_dv_t>);

    static_assert(util::is_data_v<mat_dv_t>);
    static_assert(util::is_mat_v<mat_dv_t>);

    static_assert(util::is_data_v<scl_d_t>);
    static_assert(util::is_scl_v<scl_d_t>);

    static_assert(util::is_data_v<vec_d_t>);
    static_assert(util::is_vec_v<vec_d_t>);
}

////////////////////////////////////////
// DataView: scl
////////////////////////////////////////

TEST_F(data_fixture, scl_dv_value)
{
    scl_dv_t view(&d1);
    EXPECT_DOUBLE_EQ(view.get(), d1);
}

TEST_F(data_fixture, scl_dv_size)
{
    scl_dv_t view(&d1);
    EXPECT_EQ(view.size(), 1ul);
}

TEST_F(data_fixture, scl_dv_to_ad)
{
    scl_dv_t view(&d1);
    auto expr = view.ad(ptr_pack); 
    EXPECT_DOUBLE_EQ(ad::evaluate(expr), defval1);
}

////////////////////////////////////////
// DataView: vec
////////////////////////////////////////

TEST_F(data_fixture, vec_dv_value)
{
    vec_dv_t view(values1.data(), values1.size());
    EXPECT_DOUBLE_EQ(view.get()(0), values1[0]);
    EXPECT_DOUBLE_EQ(view.get()(1), values1[1]);
    EXPECT_DOUBLE_EQ(view.get()(2), values1[2]);
}

TEST_F(data_fixture, vec_dv_size)
{
    vec_dv_t view(values1.data(), values1.size());
    EXPECT_EQ(view.size(), values1.size());
}

TEST_F(data_fixture, vec_dv_to_ad)
{
    vec_dv_t view(values1.data(), values1.size());
    auto expr = view.ad(ptr_pack); 
    Eigen::VectorXd res = ad::evaluate(expr);
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), values1[i]);
    }
}

} // namespace var
} // namespace expr
} // namespace ppl
