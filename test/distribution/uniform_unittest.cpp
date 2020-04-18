#include <autoppl/distribution/uniform.hpp>
#include <autoppl/expression/variable.hpp>

#include <cmath>
#include <array>

#include "gtest/gtest.h"

namespace ppl {

struct uniform_dist_fixture : ::testing::Test {
protected:
    Variable<double> x {0.5};
    Variable<double> y {0.1};
    Uniform<double, double> dist1 = Uniform(0., 1.);
    Uniform<double, Variable<double> > dist2 = Uniform(0., x);
    Uniform<Variable<double>, Variable<double> > dist3 = Uniform(y, x);
};

TEST_F(uniform_dist_fixture, simple_uniform) {
    EXPECT_EQ(dist1.pdf(1.1), 0.0);

    EXPECT_EQ(dist2.pdf(1.0), 0.0);
    EXPECT_EQ(dist2.pdf(0.25), 2.0);

    EXPECT_EQ(dist3.pdf(-0.1), 0.0);
    EXPECT_EQ(dist3.pdf(0.25), 2.5);
}

} // ppl