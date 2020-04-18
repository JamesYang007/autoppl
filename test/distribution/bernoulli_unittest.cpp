#include <autoppl/distribution/bernoulli.hpp>
#include <autoppl/expression/variable.hpp>

#include <cmath>
#include <array>

#include "gtest/gtest.h"

namespace ppl {

struct bernoulli_dist_fixture : ::testing::Test {
protected:
    Variable<double> x {0.6};

    Bernoulli<double> dist1 = Bernoulli(0.6);
    Bernoulli<Variable<double> > dist2 = Bernoulli(x);
};

TEST_F(bernoulli_dist_fixture, sanity_bernoulli_test) {
    EXPECT_EQ(dist1.p(), 0.6);
    EXPECT_EQ(dist2.p(), 0.6);
}

TEST_F(bernoulli_dist_fixture, simple_bernoulli) {
    EXPECT_DOUBLE_EQ(dist1.pdf(1), dist1.p());
    EXPECT_DOUBLE_EQ(dist1.pdf(1), 0.6);
    EXPECT_DOUBLE_EQ(dist1.pdf(0), 1 - dist1.p());
    EXPECT_DOUBLE_EQ(dist1.pdf(0), 0.4);
    EXPECT_DOUBLE_EQ(dist1.pdf(2), 0.0);
}

TEST_F(bernoulli_dist_fixture, bernoulli_sampling) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    
    for (int i = 0; i < 100; i++) {
        int sample = dist1.sample(gen);
        EXPECT_TRUE(sample == 0 || sample == 1);
    }
}

} // ppl