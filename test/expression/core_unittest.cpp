#include "gtest/gtest.h"
#include <autoppl/expression/core.hpp>

namespace ppl {

/*
 * Fixture for testing rv_tag with distribution.
 */
struct tag_dist_fixture : ::testing::Test
{
protected:
    rv_tag<double> x;
};

TEST_F(tag_dist_fixture, pdf_valid)
{
    auto model = (x |= uniform(0,1));
    EXPECT_EQ(model.pdf(0.000001), 1);
    EXPECT_EQ(model.pdf(0.5), 1);
    EXPECT_EQ(model.pdf(0.999999), 1);
}

TEST_F(tag_dist_fixture, pdf_invalid)
{
    auto model = (x |= uniform(0,1));
    EXPECT_EQ(model.pdf(-69), 0);
    EXPECT_EQ(model.pdf(-0.00001), 0);
    EXPECT_EQ(model.pdf(1.00001), 0);
    EXPECT_EQ(model.pdf(69), 0);
}

TEST_F(tag_dist_fixture, log_pdf_valid)
{
    auto model = (x |= uniform(0,1));
    EXPECT_EQ(model.log_pdf(0.000001), 0);
    EXPECT_EQ(model.log_pdf(0.5), 0);
    EXPECT_EQ(model.log_pdf(0.999999), 0);
}

TEST_F(tag_dist_fixture, log_pdf_invalid)
{
    auto model = (x |= uniform(0,1));
    double expected = std::numeric_limits<double>::lowest();
    EXPECT_EQ(model.log_pdf(-69), expected);
    EXPECT_EQ(model.log_pdf(-0.00001), expected);
    EXPECT_EQ(model.log_pdf(1.00001), expected);
    EXPECT_EQ(model.log_pdf(69), expected);
}

} // namespace ppl
