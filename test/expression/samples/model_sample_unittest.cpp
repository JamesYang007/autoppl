#include <array>
#include <autoppl/expr_builder.hpp>
#include <autoppl/variable.hpp>
#include <autoppl/util/dist_expr_traits.hpp>

#include <cmath>

#include "gtest/gtest.h"

namespace ppl {

struct model_sample_fixture : ::testing::Test {
   protected:
    Data<double> v1 {0.1, 0.2, 0.3, 0.4, 0.5};
    Param<double> mu, sigma;

    Param<double> w, b;
    ppl::Data<double> x{2.5, 3, 3.5, 4, 4.5, 5.};
    ppl::Data<double> y{3.5, 4, 4.5, 5, 5.5, 6.};

    ppl::Data<double> q{2.4, 3.1, 3.6, 4, 4.5, 5.};
    ppl::Data<double> r{3.5, 4, 4.4, 5.01, 5.46, 6.1};

    double tol = 1e-10;
};

TEST_F(model_sample_fixture, simple_model_test) {
    auto model = (
        mu |= uniform(-0.5, 2),
        v1 |= normal(mu, 1.0)
    );

    mu.set_value(0.0);

    EXPECT_NEAR(model.pdf(), 0.003070289574476568, tol);
    EXPECT_NEAR(model.log_pdf(), -5.785983397897518, tol);
}

TEST_F(model_sample_fixture, test_regression_pdf) {
    w.set_value(1.0);
    b.set_value(1.0);

    auto model = (w |= ppl::uniform(0, 2),
                  b |= ppl::uniform(0, 2),
                  r |= ppl::normal(q * w + b, 0.5));

    EXPECT_NEAR(model.pdf(), 0.055885938549306326, tol);
    EXPECT_NEAR(model.log_pdf(), -2.884442476988254, tol);
}

}  // namespace ppl
