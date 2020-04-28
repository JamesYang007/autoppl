#include "gtest/gtest.h"
#include <cmath>
#include <array>
#include <autoppl/expression/model/model_utils.hpp>
#include <testutil/mock_types.hpp>

namespace ppl {
namespace expr {

//////////////////////////////////////////////////////
// Model with one RV TESTS
//////////////////////////////////////////////////////

/*
 * Fixture for testing one var with distribution.
 */
struct var_dist_fixture : ::testing::Test
{
protected:
    MockParam x;
    using model_t = EqNode<MockParam, MockDistExpr>;
    model_t model = {x, MockDistExpr()};
    double val;

    void reconfigure()
    { x.set_value(val); }
};

TEST_F(var_dist_fixture, ctor)
{
    static_assert(util::assert_is_model_expr_v<model_t>);
}

TEST_F(var_dist_fixture, pdf_valid)
{
    // MockDistExpr pdf is identity function
    // so we may simply compare model.pdf() with val.
    
    val = 0.000001;
    reconfigure();
    EXPECT_EQ(model.pdf(), val);    

    val = 0.5;
    reconfigure();
    EXPECT_EQ(model.pdf(), val);

    val = 0.999999;
    reconfigure();
    EXPECT_EQ(model.pdf(), val);
}

TEST_F(var_dist_fixture, log_pdf_valid)
{
    val = 0.000001;
    reconfigure();
    EXPECT_EQ(model.log_pdf(), std::log(val));    

    val = 0.5;
    reconfigure();
    EXPECT_EQ(model.log_pdf(), std::log(val));

    val = 0.999999;
    reconfigure();
    EXPECT_EQ(model.log_pdf(), std::log(val));
}

//////////////////////////////////////////////////////
// Model with many RV (no dependencies) TESTS
//////////////////////////////////////////////////////

/*
 * Fixture for testing many vars with distributions.
 */
struct many_var_dist_fixture : ::testing::Test
{
protected:
    using value_t = double;
    using eq_t = EqNode<MockParam, MockDistExpr>;
    MockParam x, y, z, w;
    value_t xv, yv, zv, wv;

    using model_two_t = GlueNode<eq_t, eq_t>;
    model_two_t model_two = {
        {x, MockDistExpr()},
        {y, MockDistExpr()}
    };

    using model_four_t = 
        GlueNode<eq_t, 
            GlueNode<eq_t,
                GlueNode<eq_t, eq_t>
            >
        >;

    model_four_t model_four = {
        {x, MockDistExpr()},
        {
            {y, MockDistExpr()},
            {
                {z, MockDistExpr()},
                {w, MockDistExpr()}
            }
        }
    };
};

TEST_F(many_var_dist_fixture, ctor)
{
    static_assert(util::assert_is_model_expr_v<model_two_t>);
    static_assert(util::assert_is_model_expr_v<model_four_t>);
}

TEST_F(many_var_dist_fixture, two_vars_pdf)
{
    xv = 0.2; yv = 1.8;

    x.set_value(xv);
    y.set_value(yv);

    EXPECT_EQ(model_two.pdf(), xv * yv);
    EXPECT_EQ(model_two.log_pdf(), std::log(xv) + std::log(yv));
}

TEST_F(many_var_dist_fixture, four_vars_pdf)
{
    xv = 0.2; yv = 1.8; zv = 3.2; wv = 0.3;

    x.set_value(xv);
    y.set_value(yv);
    z.set_value(zv);
    w.set_value(wv);

    EXPECT_EQ(model_four.pdf(), xv * yv * zv * wv);
    EXPECT_EQ(model_four.log_pdf(), std::log(xv) + std::log(yv)
                             + std::log(zv) + std::log(wv));
}

TEST_F(many_var_dist_fixture, four_vars_traverse_count_params)
{
    int count = 0;
    model_four.traverse([&](auto&) {
            count++;
        });
    EXPECT_EQ(count, 4);
}

TEST_F(many_var_dist_fixture, four_vars_traverse_pdf)
{
    double actual = 1.;
    model_four.traverse([&](auto& model) {
            auto& var = model.get_variable();
            auto& dist = model.get_distribution();
            actual *= dist.pdf(var.get_value(0));
        });
    EXPECT_EQ(actual, model_four.pdf());
}

////////////////////////////////////////////////////////////
// get_n_params TESTS
////////////////////////////////////////////////////////////

TEST_F(many_var_dist_fixture, get_n_params_zero)
{
    using eq_node_t = EqNode<MockData, MockDistExpr>;
    static_assert(get_n_params_v<eq_node_t> == 0);
}

TEST_F(many_var_dist_fixture, get_n_params_one)
{
    using eq_node_t = EqNode<MockParam, MockDistExpr>;
    static_assert(get_n_params_v<eq_node_t> == 1);
}

TEST_F(many_var_dist_fixture, get_n_params_one_with_data)
{
    using model_t = GlueNode<
        EqNode<MockParam, MockDistExpr>,
        EqNode<MockData, MockDistExpr>
            >;
    static_assert(get_n_params_v<model_t> == 1);
}

TEST_F(many_var_dist_fixture, get_n_params_two)
{
    using model_t = GlueNode<
        EqNode<MockParam, MockDistExpr>,
        EqNode<MockParam, MockDistExpr>
            >;
    static_assert(get_n_params_v<model_t> == 2);
}


} // namespace expr
} // namespace ppl
