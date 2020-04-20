#include "gtest/gtest.h"
#include <cmath>
#include <array>
#include <autoppl/expression/model/model.hpp>
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
    MockVar x;
    using model_t = EqNode<MockVar, MockDistExpr>;
    model_t model = {x, MockDistExpr()};
    double val;

    void reconfigure()
    { x.set_value(val); }
};

TEST_F(var_dist_fixture, ctor)
{
    static_assert(util::is_model_expr_v<model_t>);
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
    MockVar x, y, z, w;
    double xv, yv, zv, wv;
    using eq_t = EqNode<MockVar, MockDistExpr>;

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
    static_assert(util::is_model_expr_v<model_two_t>);
    static_assert(util::is_model_expr_v<model_four_t>);
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
    z.set_state(MockState::data);
    model_four.traverse([&](auto& model) {
            using var_t = std::decay_t<decltype(model.get_variable())>;
            using state_t = typename util::var_traits<var_t>::state_t;
            count += (model.get_variable().get_state() == state_t::parameter);
        });
    EXPECT_EQ(count, 3);
}

TEST_F(many_var_dist_fixture, four_vars_traverse_pdf)
{
    double actual = 1.;
    model_four.traverse([&](auto& model) {
            auto& var = model.get_variable();
            auto& dist = model.get_distribution();
            actual *= dist.pdf(var);
        });
    EXPECT_EQ(actual, model_four.pdf());
}

} // namespace expr
} // namespace ppl
