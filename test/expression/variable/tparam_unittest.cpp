#include "gtest/gtest.h"
#include <testutil/base_fixture.hpp>
#include <autoppl/util/traits/var_traits.hpp>

namespace ppl {
namespace expr {
namespace var {

struct tparam_fixture: 
    base_fixture<double>,
    ::testing::Test
{
protected:
    static constexpr value_t defval1 = 1.0;
    static constexpr value_t defval2 = 2.0;
    static constexpr size_t size1 = 7;
    static constexpr size_t size2 = 17;

    info_tpack_t pack;
    size_t offset;

    value_t tol = 7e-15;

    // hypothetical tparameter values
    std::vector<value_t> values1;    
    std::vector<value_t> values2;

    tparam_fixture() 
        : pack()
        , offset(pack.off_pack.tp_offset)
        , values1(size1 + offset)
        , values2(size2 + offset)
    {
        for (size_t i = 0; i < values1.size(); ++i) {
            values1[i] = i + defval1;
        }
        for (size_t i = 0; i < values2.size(); ++i) {
            values2[i] = i + defval2;
        }
    
        ptr_pack.tp_val = values1.data();
    }
};

TEST_F(tparam_fixture, type_check)
{
    static_assert(util::is_tparam_v<scl_tpv_t>);
    static_assert(util::is_tparam_v<vec_tpv_t>);
    static_assert(util::is_tparam_v<scl_tp_t>);
    static_assert(util::is_tparam_v<vec_tp_t>);
}

////////////////////////////////////////
// TParamView: scl
////////////////////////////////////////

TEST_F(tparam_fixture, scl_tpv_value)
{
    scl_tpv_t view(&pack);
    view.bind(ptr_pack);
    EXPECT_DOUBLE_EQ(view.get(), values1[offset]);
}

TEST_F(tparam_fixture, scl_pv_size)
{
    scl_tpv_t view(&pack);
    EXPECT_EQ(view.size(), 1ul);
}

TEST_F(tparam_fixture, scl_pv_to_ad)
{
    scl_tpv_t view(&pack);

    // simply tests if gets correct elt from passed in array
    // last two parameter should be ignored
    auto elt = view.ad(ptr_pack); 
    EXPECT_DOUBLE_EQ(elt.get(), values1[offset]);

    ptr_pack.tp_val = values2.data();
    auto elt2 = view.ad(ptr_pack); 
    EXPECT_DOUBLE_EQ(elt2.get(), values2[offset]);
}

////////////////////////////////////////
// TParamView: vec
////////////////////////////////////////

TEST_F(tparam_fixture, vec_pv_value)
{
    vec_tpv_t view(&pack, size1);
    view.bind(ptr_pack);
    for (size_t i = 0; i < size1; ++i) {
        EXPECT_DOUBLE_EQ(view.get()(i), values1[offset+i]);
    }
}

TEST_F(tparam_fixture, vec_pv_size)
{
    vec_tpv_t view(&pack, size1);
    EXPECT_EQ(view.size(), size1);
}

TEST_F(tparam_fixture, vec_pv_to_ad)
{
    vec_tpv_t view(&pack, size1);

    auto elt = view.ad(ptr_pack);

    for (size_t i = 0; i < size1; ++i) {
        EXPECT_DOUBLE_EQ(elt.get()(i), values1[offset+i]);
    }
}

} // namespace var
} // namespace expr
} // namespace ppl
