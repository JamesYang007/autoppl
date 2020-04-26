#include <array>
#include <autoppl/expr_builder.hpp>
#include <autoppl/variable.hpp>
#include <autoppl/util/dist_expr_traits.hpp>

#include <cmath>

#include "gtest/gtest.h"

namespace ppl {

struct normal_fixture : ::testing::Test {
   protected:
    Variable<double> v1 {0.1, 0.2, 0.3, 0.4, 0.5};
    Variable<double> x {0.0};
    Variable<double> y {0.0};

    double tol = 1e-15;
};

TEST_F(normal_fixture, normal_check_pdf) {
    auto dist1 = normal(0., 1.);

    EXPECT_NEAR(util::pdf(v1, dist1), 0.0076757239361914193, tol);
    EXPECT_NEAR(util::log_pdf(v1, dist1), -4.869692666023363, tol);

    auto dist2 = normal(x, 1.);

    EXPECT_NEAR(util::pdf(v1, dist2), 0.0076757239361914193, tol);
    EXPECT_NEAR(util::log_pdf(v1, dist2), -4.869692666023363, tol);

    x.set_value(0.1);
    y.set_value(-0.1);
    auto dist3 = normal(x + y, 1.);

    EXPECT_NEAR(util::pdf(v1, dist3), 0.0076757239361914193, tol);
    EXPECT_NEAR(util::log_pdf(v1, dist3), -4.869692666023363, tol);
}

}  // namespace ppl
