#include "gtest/gtest.h"
#include <testutil/base_fixture.hpp>
#include <autoppl/expression/variable/param.hpp>
#include <autoppl/expression/constraint/pos_def.hpp>
#include <autoppl/util/traits/var_traits.hpp>

namespace ppl {
namespace expr {
namespace var {

struct param_fixture: 
    base_fixture<double>,
    ::testing::Test
{
protected:
    using pos_def_t = constraint::PosDef;
    using pos_def_mat_pv_t = ParamView<value_t, mat, pos_def_t>;

    static constexpr value_t defval1 = 1.0;
    static constexpr value_t defval2 = 2.0;
    static constexpr size_t size1 = 7;
    static constexpr size_t size2 = 17;

    info_pack_t pack;
    size_t offset;

    value_t tol = 7e-15;

    // hypothetical parameter values
    std::vector<value_t> values1;    
    std::vector<value_t> values2;

    param_fixture() 
        : pack()
        , offset(pack.off_pack.uc_offset)
        , values1(size1 + offset)
        , values2(size2 + offset)
    {
        for (size_t i = 0; i < values1.size(); ++i) {
            values1[i] = i + defval1;
        }
        for (size_t i = 0; i < values2.size(); ++i) {
            values2[i] = i + defval2;
        }
    
        ptr_pack.uc_val = values1.data();
    }
};

TEST_F(param_fixture, type_check)
{
    static_assert(util::is_param_v<scl_pv_t>);
    static_assert(util::is_param_v<vec_pv_t>);
    static_assert(util::is_param_v<scl_p_t>);
    static_assert(util::is_param_v<vec_p_t>);
}

////////////////////////////////////////
// ParamView: scl
////////////////////////////////////////

TEST_F(param_fixture, scl_pv_value)
{
    scl_pv_t view(&pack);
    view.bind(ptr_pack);
    EXPECT_DOUBLE_EQ(view.get(), values1[offset]);
}

TEST_F(param_fixture, scl_pv_size)
{
    scl_pv_t view(&pack);
    EXPECT_EQ(view.size(), 1ul);
}

TEST_F(param_fixture, scl_pv_to_ad)
{
    scl_pv_t view(&pack);

    // simply tests if gets correct elt from passed in array
    // last two parameter should be ignored
    auto elt = view.ad(ptr_pack); 
    EXPECT_DOUBLE_EQ(elt.get(), values1[offset]);

    ptr_pack.uc_val = values2.data();
    auto elt2 = view.ad(ptr_pack); 
    EXPECT_DOUBLE_EQ(elt2.get(), values2[offset]);
}

////////////////////////////////////////
// ParamView: vec
////////////////////////////////////////

TEST_F(param_fixture, vec_pv_value)
{
    vec_pv_t view(&pack, size1);
    view.bind(ptr_pack);
    for (size_t i = 0; i < size1; ++i) {
        EXPECT_DOUBLE_EQ(view.get()(i), values1[offset+i]);
    }
}

TEST_F(param_fixture, vec_pv_size)
{
    vec_pv_t view(&pack, size1);
    EXPECT_EQ(view.size(), size1);
}

TEST_F(param_fixture, vec_pv_to_ad)
{
    vec_pv_t view(&pack, size1);

    auto elt = view.ad(ptr_pack);

    for (size_t i = 0; i < size1; ++i) {
        EXPECT_DOUBLE_EQ(elt.get()(i), values1[offset+i]);
    }
}

////////////////////////////////////////
// ParamView: pos-def mat
////////////////////////////////////////
TEST_F(param_fixture, pos_def_mat_pv_transform)
{
    pos_def_mat_pv_t S(&pack, 2, 2);    
    std::vector<value_t> c(100, 0); // obscene amount
    std::vector<size_t> v(100, 0); // obscene amount
    Eigen::Map<Eigen::VectorXd> mp(values1.data(), values1.size());
    Eigen::VectorXd orig = mp;

    ptr_pack.c_val = c.data();
    ptr_pack.v_val = v.data();
    S.activate_refcnt();
    S.bind(ptr_pack);
    S.eval();
    S.inv_eval();
    for (int i = 0; i < mp.size(); ++i) {
        EXPECT_NEAR(mp(i), orig(i), tol);
    }
}

} // namespace var
} // namespace expr
} // namespace ppl
