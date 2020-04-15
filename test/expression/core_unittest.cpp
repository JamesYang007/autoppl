#include "gtest/gtest.h"
#include <autoppl/expression/core.hpp>

namespace ppl {

//////////////////////////////////////////////////////
// Model with one RV TESTS
//////////////////////////////////////////////////////

/*
 * Fixture for testing one rv_tag with distribution.
 */
struct tag_dist_fixture : ::testing::Test
{
protected:
    rv_tag<double> x;
    using model_t = std::decay_t<decltype(x |= uniform(0,1))>;
    model_t model = (x |= uniform(0,1));

    void reconfigure(double val)
    {
        x.set_value(val);
        model.update();
    }
};

TEST_F(tag_dist_fixture, pdf_valid)
{
    reconfigure(0.000001);
    EXPECT_EQ(model.pdf(), 1);
    reconfigure(0.5);
    EXPECT_EQ(model.pdf(), 1);
    reconfigure(0.999999);
    EXPECT_EQ(model.pdf(), 1);
}

TEST_F(tag_dist_fixture, pdf_invalid)
{
    reconfigure(-69);
    EXPECT_EQ(model.pdf(), 0);
    reconfigure(-0.000001);
    EXPECT_EQ(model.pdf(), 0);
    reconfigure(1.000001);
    EXPECT_EQ(model.pdf(), 0);
    reconfigure(69);
    EXPECT_EQ(model.pdf(), 0);
}

TEST_F(tag_dist_fixture, log_pdf_valid)
{
    reconfigure(0.000001);
    EXPECT_EQ(model.log_pdf(), 0);
    reconfigure(0.5);
    EXPECT_EQ(model.log_pdf(), 0);
    reconfigure(0.999999);
    EXPECT_EQ(model.log_pdf(), 0);
}

TEST_F(tag_dist_fixture, log_pdf_invalid)
{
    double expected = std::numeric_limits<double>::lowest();
    reconfigure(-69);
    EXPECT_EQ(model.log_pdf(), expected);
    reconfigure(-0.000001);
    EXPECT_EQ(model.log_pdf(), expected);
    reconfigure(1.000001);
    EXPECT_EQ(model.log_pdf(), expected);
    reconfigure(69);
    EXPECT_EQ(model.log_pdf(), expected);
}

//////////////////////////////////////////////////////
// Model with many RV (no dependencies) TESTS
//////////////////////////////////////////////////////

struct many_tag_dist_fixture : ::testing::Test
{
protected:
    rv_tag<double> x, y, z, w;
};

TEST_F(many_tag_dist_fixture, two_tags)
{
    auto model = (
        x |= uniform(0, 1),
        y |= uniform(0, 2)
    );

    auto reconfigure = [&](double xv, double yv) {
        x.set_value(xv);
        y.set_value(yv);
        model.update();
    };

    // both within range
    reconfigure(0.2, 1.8);
    EXPECT_EQ(model.pdf(), 0.5);

    // first out of range
    reconfigure(-0.0005, 0.99999);
    EXPECT_EQ(model.pdf(), 0);

    // second out of range
    reconfigure(0.5142, 2.0000123);
    EXPECT_EQ(model.pdf(), 0);

    // both out of range
    reconfigure(-10., 69.);
    EXPECT_EQ(model.pdf(), 0);
}

} // namespace ppl
