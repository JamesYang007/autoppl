#include <cmath>
#include <array>
#include "gtest/gtest.h"
#include <autoppl/expression/model.hpp>

namespace ppl {
namespace expr {

//////////////////////////////////////////////////////
// Model with one RV TESTS
//////////////////////////////////////////////////////

/*
 * Mock state class for testing purposes.
 */
enum class MockState {
    data,
    parameter
};

/*
 * Mock var object for testing purposes.
 * Must meet some of the requirements of actual var types.
 */
struct MockVar 
{
    using value_t = double;
    using pointer_t = double*;
    using state_t = MockState;

    void set_value(double val) { value_ = val; }  
    value_t get_value() const { return value_; }  

    void set_state(state_t state) { state_ = state; }
    state_t get_state() const { return state_; }

    operator value_t() { return value_; }

private:
    double value_;
    state_t state_ = state_t::parameter;
};

/*
 * Mock distribution object for testing purposes.
 * Must meet some of the requirements of actual distribution types.
 */
struct MockDist
{
    using value_t = double;
    using dist_value_t = double;

    // Mock pdf - identity function.
    double pdf(double x) const
    { return x; }

    // Mock log_pdf - log(pdf(x)). 
    double log_pdf(double x) const
    { return std::log(pdf(x)); }
};

/*
 * Fixture for testing one var with distribution.
 */
struct var_dist_fixture : ::testing::Test
{
protected:
    MockVar x;
    using model_t = EqNode<MockVar, MockDist>;
    model_t model = {x, MockDist()};
    double val;

    void reconfigure(double val)
    {
        x.set_value(val);
    }
};

TEST_F(var_dist_fixture, pdf_valid)
{
    // MockDist pdf is identity function
    // so we may simply compare model.pdf() with val.
    
    val = 0.000001;
    reconfigure(val);
    EXPECT_EQ(model.pdf(), val);    

    val = 0.5;
    reconfigure(val);
    EXPECT_EQ(model.pdf(), val);

    val = 0.999999;
    reconfigure(val);
    EXPECT_EQ(model.pdf(), val);
}

TEST_F(var_dist_fixture, pdf_invalid)
{
    val = 0.000004123;
    reconfigure(val);
    EXPECT_EQ(model.pdf(), val);

    val = 0.55555;
    reconfigure(val);
    EXPECT_EQ(model.pdf(), val);

    val = 5.234424231;
    reconfigure(val);
    EXPECT_EQ(model.pdf(), val);

    val = 69;
    reconfigure(val);
    EXPECT_EQ(model.pdf(), val);
}

TEST_F(var_dist_fixture, log_pdf_valid)
{
    val = 0.000001;
    reconfigure(val);
    EXPECT_EQ(model.log_pdf(), std::log(val));    

    val = 0.5;
    reconfigure(val);
    EXPECT_EQ(model.log_pdf(), std::log(val));

    val = 0.999999;
    reconfigure(val);
    EXPECT_EQ(model.log_pdf(), std::log(val));
}

TEST_F(var_dist_fixture, log_pdf_invalid)
{
    val = 0.000004123;
    reconfigure(val);
    EXPECT_EQ(model.log_pdf(), std::log(val));

    val = 0.55555;
    reconfigure(val);
    EXPECT_EQ(model.log_pdf(), std::log(val));

    val = 5.234424231;
    reconfigure(val);
    EXPECT_EQ(model.log_pdf(), std::log(val));

    val = 69;
    reconfigure(val);
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
    using eq_t = EqNode<MockVar, MockDist>;

    using model_two_t = GlueNode<eq_t, eq_t>;
    model_two_t model_two = {
        {x, MockDist()},
        {y, MockDist()}

    };

    using model_four_t = 
        GlueNode<eq_t, 
            GlueNode<eq_t,
                GlueNode<eq_t, eq_t>
            >
        >;

    model_four_t model_four = {
        {x, MockDist()},
        {
            {y, MockDist()},
            {
                {z, MockDist()},
                {w, MockDist()}
            }
        }
    };
};

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
            using state_t = typename var_traits<var_t>::state_t;
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
