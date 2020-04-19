#include <autoppl/distribution/uniform.hpp>
#include <autoppl/expression/variable.hpp>

#include <cmath>
#include <array>

#include "gtest/gtest.h"

namespace ppl {
namespace dist {

struct uniform_dist_fixture : ::testing::Test {
protected:
    Variable<double> x {0.5};
    Variable<double> y {0.1};
    Uniform<double, double> dist1 = Uniform(0., 1.);
    Uniform<double, Variable<double> > dist2 = Uniform(0., x);
    Uniform<Variable<double>, Variable<double> > dist3 = Uniform(y, x);
};

TEST_F(uniform_dist_fixture, sanity_uniform_test) {
    EXPECT_EQ(dist1.min(), 0.0);
    EXPECT_EQ(dist1.max(), 1.0);

    EXPECT_EQ(dist2.min(), 0.0);
    EXPECT_EQ(dist2.max(), 0.5);

    EXPECT_EQ(dist3.min(), 0.1);
    EXPECT_EQ(dist3.max(), 0.5);
}

TEST_F(uniform_dist_fixture, simple_uniform) {
    EXPECT_DOUBLE_EQ(dist1.pdf(1.1), 0.0);
    EXPECT_DOUBLE_EQ(dist1.pdf(1.0), 0.0);

    EXPECT_DOUBLE_EQ(dist2.pdf(0.25), 2.0);
    EXPECT_DOUBLE_EQ(dist2.pdf(-0.1), 0.0);

    EXPECT_DOUBLE_EQ(dist3.pdf(0.25), 2.5);
}

TEST_F(uniform_dist_fixture, uniform_sampling) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    
    for (int i = 0; i < 100; i++) {
        double sample = dist1.sample(gen);
        EXPECT_GT(sample, 0.0);
        EXPECT_LT(sample, 1.0);
    }
}

} // namespace dist
} // namespace ppl
