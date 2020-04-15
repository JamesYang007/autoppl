#include <cmath>
#include "gtest/gtest.h"
#include <autoppl/expression/model.hpp>

namespace ppl {

//////////////////////////////////////////////////////
// Model with one RV TESTS
//////////////////////////////////////////////////////

/*
 * Mock tag object for testing purposes.
 * Must meet some of the requirements of actual tag types.
 */
struct MockTag
{
    using value_t = double;
    using pointer_t = double*;
    using state_t = void;

    void set_value(double val) { value_ = val; }  
    double get_value() const { return value_; }  

private:
    double value_;
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
 * Fixture for testing one rv_tag with distribution.
 */
struct tag_dist_fixture : ::testing::Test
{
protected:
    MockTag x;
    using model_t = std::decay_t<decltype(x |= MockDist())>;
    model_t model = (x |= MockDist());
    double val;

    void reconfigure(double val)
    {
        x.set_value(val);
        model.update();
    }
};

TEST_F(tag_dist_fixture, pdf_valid)
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

TEST_F(tag_dist_fixture, pdf_invalid)
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

TEST_F(tag_dist_fixture, log_pdf_valid)
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

TEST_F(tag_dist_fixture, log_pdf_invalid)
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

struct many_tag_dist_fixture : ::testing::Test
{
protected:
    MockTag x, y, z, w;
    double xv, yv, zv, wv;
};

TEST_F(many_tag_dist_fixture, two_tags)
{
    auto model = (
        x |= MockDist(),
        y |= MockDist()
    );

    auto reconfigure = [&](double xv, double yv) {
        x.set_value(xv);
        y.set_value(yv);
        model.update();
    };

    // both within range
    xv = 0.2; yv = 1.8;
    reconfigure(xv, yv);
    EXPECT_EQ(model.pdf(), xv * yv);
    EXPECT_EQ(model.log_pdf(), std::log(xv) + std::log(yv));

    // first out of range
    xv = 0.000542431; yv = 0.99999;
    reconfigure(xv, yv);
    EXPECT_EQ(model.pdf(), xv * yv);
    EXPECT_EQ(model.log_pdf(), std::log(xv) + std::log(yv));

    // second out of range
    xv = 0.5142; yv = 2.0000123;
    reconfigure(xv, yv);
    EXPECT_EQ(model.pdf(), xv * yv);
    EXPECT_EQ(model.log_pdf(), std::log(xv) + std::log(yv));

    // both out of range
    xv = 14.3; yv = 69.;
    reconfigure(xv, yv);
    EXPECT_EQ(model.pdf(), xv * yv);
    EXPECT_EQ(model.log_pdf(), std::log(xv) + std::log(yv));
}

} // namespace ppl
