#include <autoppl/expression/distribution/discrete.hpp>

#include <cmath>
#include <array>

#include "gtest/gtest.h"

namespace ppl {
namespace expr {

struct discrete_dist_fixture : ::testing::Test {
protected:
    std::vector<double> weights {0.1, 0.2, 0.3, 0.4};
    Discrete<double> dist1 =  {1.0, 2.0, 3.0, 4.0};
    Discrete<double> dist2;
    Discrete<double> dist3 = Discrete(weights.begin(), weights.end());
};

TEST_F(discrete_dist_fixture, default_cstor_test){
    EXPECT_DOUBLE_EQ(dist2.weights(0), 1.0);
    EXPECT_DOUBLE_EQ(dist2.pdf(0), 1.0);
}

TEST_F(discrete_dist_fixture, sanity_Discrete_test) {
    EXPECT_DOUBLE_EQ(dist1.weights(0), weights[0]);
    EXPECT_DOUBLE_EQ(dist1.weights(1), weights[1]);
    EXPECT_DOUBLE_EQ(dist1.weights(2), weights[2]);
    EXPECT_DOUBLE_EQ(dist1.weights(3), weights[3]);
}

TEST_F(discrete_dist_fixture, simple_Discrete) {
    EXPECT_DOUBLE_EQ(dist1.pdf(0), dist1.weights(0));
    EXPECT_DOUBLE_EQ(dist1.pdf(1), dist1.weights(1));
    EXPECT_DOUBLE_EQ(dist1.pdf(2), dist1.weights(2));
    EXPECT_DOUBLE_EQ(dist1.pdf(3), dist1.weights(3));
    
}

TEST_F(discrete_dist_fixture, Discrete_sampling) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    
    for (int i = 0; i < 100; i++) {
        int sample = dist1.sample(gen);
        EXPECT_TRUE(sample == 0 || sample == 1 || sample == 2 || sample == 3);
    }
}

TEST_F(discrete_dist_fixture, sanity_Discrete_iter_test) {
    EXPECT_DOUBLE_EQ(dist3.weights(0), weights[0]);
    EXPECT_DOUBLE_EQ(dist3.weights(1), weights[1]);
    EXPECT_DOUBLE_EQ(dist3.weights(2), weights[2]);
    EXPECT_DOUBLE_EQ(dist3.weights(3), weights[3]);
}

TEST_F(discrete_dist_fixture, simple_Discrete_iter) {
    EXPECT_DOUBLE_EQ(dist3.pdf(0), dist3.weights(0));
    EXPECT_DOUBLE_EQ(dist3.pdf(1), dist3.weights(1));
    EXPECT_DOUBLE_EQ(dist3.pdf(2), dist3.weights(2));
    EXPECT_DOUBLE_EQ(dist3.pdf(3), dist3.weights(3));
    
}

TEST_F(discrete_dist_fixture, Discrete_sampling_iter) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    
    for (int i = 0; i < 100; i++) {
        int sample = dist3.sample(gen);
        EXPECT_TRUE(sample == 0 || sample == 1 || sample == 2 || sample == 3);
    }
}
} // namespace expr
} // ppl
