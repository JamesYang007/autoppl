#include <autoppl/distribution/normal.hpp>
#include <autoppl/expression/variable.hpp>

#include <cmath>
#include <array>

#include "gtest/gtest.h"

namespace ppl {
namespace dist {

struct normal_dist_fixture : ::testing::Test {
protected:
    Variable<double> mu {0.};
    Variable<double> sigma {1.};
    Normal<double, double> dist1 = Normal(0., 1.);
    Normal<Variable<double>, Variable<double> > dist2 = Normal(mu, sigma);
};

TEST_F(normal_dist_fixture, sanity_normal_test) {
    EXPECT_EQ(dist1.mean(), 0.0);
    EXPECT_EQ(dist1.var(), 1.0);

    EXPECT_EQ(dist2.mean(), 0.0);
    EXPECT_EQ(dist2.var(), 1.0);
}

TEST_F(normal_dist_fixture, simple_gaussian) {
    EXPECT_DOUBLE_EQ(dist1.pdf(0.0), 0.3989422804014327);
    EXPECT_DOUBLE_EQ(dist1.pdf(-0.5), 0.3520653267642995);
    EXPECT_DOUBLE_EQ(dist1.pdf(4), 0.00013383022576488537);

    EXPECT_DOUBLE_EQ(dist1.log_pdf(0.0), std::log(dist1.pdf(0.0)));
    EXPECT_DOUBLE_EQ(dist1.log_pdf(-0.5), std::log(dist1.pdf(-0.5)));
    EXPECT_DOUBLE_EQ(dist1.log_pdf(4), std::log(dist1.pdf(4)));


    EXPECT_DOUBLE_EQ(dist2.pdf(0.0), 0.3989422804014327);
    EXPECT_DOUBLE_EQ(dist2.pdf(-0.5), 0.3520653267642995);
    EXPECT_DOUBLE_EQ(dist2.pdf(4), 0.00013383022576488537);

    EXPECT_DOUBLE_EQ(dist2.log_pdf(0.0), std::log(dist2.pdf(0.0)));
    EXPECT_DOUBLE_EQ(dist2.log_pdf(-0.5), std::log(dist2.pdf(-0.5)));
    EXPECT_DOUBLE_EQ(dist2.log_pdf(4), std::log(dist1.pdf(4)));
}

} // namespace dist
} // namespace ppl
