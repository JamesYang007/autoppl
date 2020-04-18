#include <autoppl/expression/uniform.hpp>
#include <autoppl/expression/rv_tag.hpp>
#include <autoppl/expression/model.hpp>

#include <cmath>
#include <array>

#include "gtest/gtest.h"

namespace ppl {

struct uniform_dist_fixture : ::testing::Test {
protected:
    rv_tag<double> x {0.5};
    rv_tag<double> y {0.1};
    Uniform<double, double> dist1 = Uniform(0., 1.);
    Uniform<double, rv_tag<double> > dist2 = Uniform(0., x);
    Uniform<rv_tag<double>, rv_tag<double> > dist3 = Uniform(y, x);
};

TEST_F(uniform_dist_fixture, simple_uniform) {
    EXPECT_EQ(dist1.pdf(1.1), 0.0);
    EXPECT_EQ(dist2.pdf(1.0), 0.0);
    EXPECT_EQ(dist2.pdf(0.25), 2.0);
    EXPECT_EQ(dist3.pdf(-0.1), 0.0);
    EXPECT_EQ(dist3.pdf(0.25), 2.5);
}

} // ppl