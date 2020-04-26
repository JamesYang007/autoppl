#include <array>
#include <autoppl/expr_builder.hpp>
#include <autoppl/variable.hpp>
#include <autoppl/util/dist_expr_traits.hpp>

#include <cmath>

#include "gtest/gtest.h"

namespace ppl {

struct model_fixture : ::testing::Test {
   protected:
    Variable<double> v1 {0.1, 0.2, 0.3, 0.4, 0.5};
    Variable<double> mu;
    Variable<double> sigma;

    double tol = 1e-15;
};

TEST_F(model_fixture, simple_model_test) {
    auto model = (
        mu |= uniform(-0.5, 2),
        v1 |= normal(mu, 1.0)
    );

    mu.set_value(0.0);

    EXPECT_NEAR(model.pdf(), 0.003070289574476568, tol);
    EXPECT_NEAR(model.log_pdf(), -5.785983397897518, tol);
}

}  // namespace ppl
