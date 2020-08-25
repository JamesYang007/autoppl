#include "gtest/gtest.h"
#include <testutil/base_fixture.hpp>
#include <cmath>
#include <array>
#include <autoppl/expression/model/bar_eq.hpp>
#include <autoppl/expression/model/glue.hpp>
#include <autoppl/expression/distribution/normal.hpp>
#include <autoppl/math/density.hpp>
#include <autoppl/expression/variable/constant.hpp>
#include <autoppl/util/traits/var_traits.hpp>

namespace ppl {
namespace expr {

//////////////////////////////////////////////////////
// Model with one RV TESTS
//////////////////////////////////////////////////////

/*
 * Fixture for testing one var with distribution.
 */
struct model_fixture: 
    base_fixture<double>,
    ::testing::Test
{
protected:
    using param_t = scl_p_t;
    using dist_t = dist::Normal<scl_c_t, scl_c_t>;
    using eq_t = model::BarEqNode<scl_pv_t, dist_t>;

    param_t x;
    dist_t dist;
    eq_t model;

    model_fixture()
        : x()
        , dist(0., 1.)
        , model(x, dist)
    {
        offset_pack_t offset;
        x.activate(offset);
        val_buf.resize(100);
        ptr_pack.uc_val = val_buf.data();
        model.bind(ptr_pack);
    }
};

TEST_F(model_fixture, type_check)
{
    static_assert(util::is_model_expr_v<eq_t>);
}

TEST_F(model_fixture, eq_pdf_valid)
{
    val_buf[0] = 1.5;
    EXPECT_DOUBLE_EQ(model.pdf(), math::normal_pdf(val_buf[0], 0, 1)); 
}

TEST_F(model_fixture, eq_log_pdf_valid)
{
    val_buf[0] = 2.;
    EXPECT_DOUBLE_EQ(model.log_pdf(), math::normal_log_pdf(val_buf[0], 0, 1));
}

//////////////////////////////////////////////////////
// Model with many RV (no dependencies) TESTS
//////////////////////////////////////////////////////

/*
 * Fixture for testing many vars with distributions.
 */
struct many_model_fixture: 
    base_fixture<double>,
    ::testing::Test
{
protected:
    using dist_t = dist::Normal<scl_c_t, scl_c_t>;
    using eq_t = model::BarEqNode<scl_pv_t, dist_t>;
    scl_p_t x;
    scl_p_t y;
    scl_p_t z;

    using model_two_t = model::GlueNode<eq_t, eq_t>;
    model_two_t model_two = {
        {x, {0,1}},
        {y, {0,1}}
    };

    using model_three_t = model::GlueNode<eq_t, model_two_t>;
    model_three_t model_three = {
        {x, {0,1}},
        {
            {y, {0,1}},
            {z, {0,1}}
        }
    };

    many_model_fixture()
    {
        offset_pack_t offset;
        x.activate(offset);
        y.activate(offset);
        z.activate(offset);
        val_buf.resize(100);
        val_buf[0] = 0.2;
        val_buf[1] = 1.8;
        val_buf[2] = 0.32;

        ptr_pack.uc_val = val_buf.data();
        model_two.bind(ptr_pack);
        model_three.bind(ptr_pack);
    }
};

TEST_F(many_model_fixture, type_check)
{
    static_assert(util::is_model_expr_v<model_two_t>);
    static_assert(util::is_model_expr_v<model_three_t>);
}

TEST_F(many_model_fixture, two_vars_pdf)
{
    EXPECT_DOUBLE_EQ(model_two.pdf(),
                     math::normal_pdf(val_buf[0], 0, 1) *
                     math::normal_pdf(val_buf[1], 0, 1));
    EXPECT_DOUBLE_EQ(model_two.log_pdf(), 
                     math::normal_log_pdf(val_buf[0], 0, 1) +
                     math::normal_log_pdf(val_buf[1], 0, 1));
}

TEST_F(many_model_fixture, three_vars_pdf)
{
    EXPECT_DOUBLE_EQ(model_three.pdf(),
                     math::normal_pdf(val_buf[0], 0, 1) *
                     math::normal_pdf(val_buf[1], 0, 1) *
                     math::normal_pdf(val_buf[2], 0, 1));
    EXPECT_DOUBLE_EQ(model_three.log_pdf(), 
                     math::normal_log_pdf(val_buf[0], 0, 1) +
                     math::normal_log_pdf(val_buf[1], 0, 1) +
                     math::normal_log_pdf(val_buf[2], 0, 1));
}

TEST_F(many_model_fixture, three_vars_traverse_pdf)
{
    double actual = 1.;
    model_three.traverse([&](auto& eq) {
            auto& var = eq.get_variable();
            auto& dist = eq.get_distribution();
            actual *= dist.pdf(var);
        });
    EXPECT_DOUBLE_EQ(actual, model_three.pdf());
}

} // namespace expr
} // namespace ppl
