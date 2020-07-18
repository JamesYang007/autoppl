#include "gtest/gtest.h"
#include <cmath>
#include <array>
#include <autoppl/expression/model/eq_node.hpp>
#include <autoppl/expression/model/glue_node.hpp>
#include <autoppl/util/traits/var_traits.hpp>
#include <autoppl/util/traits/mock_types.hpp>

namespace ppl {
namespace expr {

//////////////////////////////////////////////////////
// Model with one RV TESTS
//////////////////////////////////////////////////////

/*
 * Fixture for testing one var with distribution.
 */
struct model_fixture : ::testing::Test
{
protected:
    using param_t = MockParam;
    using value_t = typename util::param_traits<param_t>::value_t;
    using dist_t = MockDistExpr;
    using eq_t = EqNode<param_t, dist_t>;
};

TEST_F(model_fixture, type_check)
{
    static_assert(util::is_model_expr_v<eq_t>);
}

TEST_F(model_fixture, eq_pdf_valid)
{
    param_t x(3.);
    dist_t d(0.5);
    eq_t model(x, d);
    value_t val = 1.5;
    // parameter ignored (arbitrary)
    EXPECT_DOUBLE_EQ(model.pdf(0), val); 
}

TEST_F(model_fixture, eq_log_pdf_valid)
{
    param_t x(5.);
    dist_t d(1.32);
    eq_t model(x, d);
    value_t val = std::log(5. * 1.32);
    // parameter ignored (arbitrary)
    EXPECT_DOUBLE_EQ(model.log_pdf(0), val);
}

//////////////////////////////////////////////////////
// Model with many RV (no dependencies) TESTS
//////////////////////////////////////////////////////

/*
 * Fixture for testing many vars with distributions.
 */
struct many_model_fixture : ::testing::Test
{
protected:
    using value_t = double;
    using eq_t = EqNode<MockParam, MockDistExpr>;
    value_t xv = 0.2;
    value_t yv = 1.8; 
    value_t zv = 0.32;
    value_t xd = 1.5;
    value_t yd = 1.523;
    value_t zd = 0.00132;
    MockParam x = xv;
    MockParam y = yv;
    MockParam z = zv;

    using model_two_t = GlueNode<eq_t, eq_t>;
    model_two_t model_two = {
        {x, MockDistExpr(xd)},
        {y, MockDistExpr(yd)}
    };

    using model_three_t = 
        GlueNode<eq_t, GlueNode<eq_t, eq_t>>;

    model_three_t model_three = {
        {x, MockDistExpr(xd)},
        {
            {y, MockDistExpr(yd)},
            {z, MockDistExpr(zd)}
        }
    };
};

TEST_F(many_model_fixture, type_check)
{
    static_assert(util::is_model_expr_v<model_two_t>);
    static_assert(util::is_model_expr_v<model_three_t>);
}

TEST_F(many_model_fixture, two_vars_pdf)
{
    EXPECT_DOUBLE_EQ(model_two.pdf(0), xv * xd * yv * yd);
    EXPECT_DOUBLE_EQ(model_two.log_pdf(0), std::log(xv*xd) + std::log(yv*yd));
}

TEST_F(many_model_fixture, three_vars_pdf)
{
    EXPECT_DOUBLE_EQ(model_three.pdf(0), xv * xd * yv * yd * zv * zd);
    EXPECT_DOUBLE_EQ(model_three.log_pdf(0), 
              std::log(xv*xd) + std::log(yv*yd) + std::log(zv*zd));
}

TEST_F(many_model_fixture, three_vars_traverse_pdf)
{
    double actual = 1.;
    model_three.traverse([&](auto& eq) {
            auto& var = eq.get_variable();
            auto& dist = eq.get_distribution();
            actual *= dist.pdf(var, 0);
        });
    EXPECT_DOUBLE_EQ(actual, model_three.pdf(0));
}

} // namespace expr
} // namespace ppl
