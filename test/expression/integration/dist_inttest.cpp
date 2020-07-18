#include "gtest/gtest.h"
#include <cmath>
#include <array>
#include <autoppl/expression/expr_builder.hpp>
#include <autoppl/expression/variable/data.hpp>
#include <autoppl/util/traits/dist_expr_traits.hpp>

namespace ppl {

struct normal_integration_fixture : ::testing::Test {
protected:
    using value_t = double;
    using param_t = Param<value_t, ppl::vec>;
    using data_t = Data<value_t, ppl::vec>;
    using pview_t = typename param_t::base_t;

    data_t v1 {0.1, 0.2, 0.3, 0.4, 0.5};
    std::array<value_t, 2> pvalues = {0.1, -0.1};
    param_t x = 2;

    value_t tol = 1e-15;

    normal_integration_fixture()
    {
        // manually set offset
        // in real-use case, user will call an initialization function
        pview_t x_view = x;
        x_view.set_offset(0);
    }
};

TEST_F(normal_integration_fixture, normal_pdfs) {
    
    auto dist1 = normal(0., 1.);

    EXPECT_NEAR(dist1.pdf(v1, pvalues), 0.007675723936191419, tol);
    EXPECT_NEAR(dist1.log_pdf(v1, pvalues), -4.869692666023363, tol);

    auto dist2 = normal(x[0], 1.);

    pvalues[0] = 0.;
    EXPECT_NEAR(dist2.pdf(v1, pvalues), 0.0076757239361914193, tol);
    EXPECT_NEAR(dist2.log_pdf(v1, pvalues), -4.869692666023363, tol);

    auto dist3 = normal(x[0] + x[1], 1.);

    pvalues[0] = 0.1;
    EXPECT_NEAR(dist3.pdf(v1, pvalues), 0.0076757239361914193, tol);
    EXPECT_NEAR(dist3.log_pdf(v1, pvalues), -4.869692666023363, tol);
}

}  // namespace ppl
