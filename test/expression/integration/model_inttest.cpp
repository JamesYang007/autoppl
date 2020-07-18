#include "gtest/gtest.h"
#include <cmath>
#include <array>
#include <autoppl/expression/expr_builder.hpp>
#include <autoppl/util/traits/dist_expr_traits.hpp>

namespace ppl {

struct model_integration_fixture : ::testing::Test {
protected:
    using value_t = double;
    using data_t = Data<value_t, ppl::vec>;
    using param_t = Param<value_t, ppl::scl>;
    using pview_t = typename param_t::base_t;

    value_t tol = 1e-15;

    data_t v1 {0.1, 0.2, 0.3, 0.4, 0.5};
    param_t mu, sigma, w, b;
    std::array<value_t, 4> pvalues;

    data_t x{2.5, 3, 3.5, 4, 4.5, 5.};
    data_t y{3.5, 4, 4.5, 5, 5.5, 6.};
    data_t q{2.4, 3.1, 3.6, 4, 4.5, 5.};
    data_t r{3.5, 4, 4.4, 5.01, 5.46, 6.1};

    model_integration_fixture()
    {
        // manually set offset
        // in real-use case, user will call an initialization function
        pview_t mu_view = mu;
        pview_t sigma_view = sigma;
        pview_t w_view = w;
        pview_t b_view = b;

        auto next = mu_view.set_offset(0);
        next = sigma_view.set_offset(next);
        next = w_view.set_offset(next);
        b_view.set_offset(next);
    }
};

TEST_F(model_integration_fixture, simple_model_pdfs) {
    auto model = (
        mu |= uniform(-0.5, 2.),
        v1 |= normal(mu, 1.0)
    );

    pvalues[0] = 0.0;

    EXPECT_NEAR(model.pdf(pvalues), 0.003070289574476568, tol);
    EXPECT_NEAR(model.log_pdf(pvalues), -5.785983397897518, tol);
}

TEST_F(model_integration_fixture, regression_pdfs) {
    pvalues[2] = 1.0;
    pvalues[3] = 1.0;

    auto model = (w |= ppl::uniform(0., 2.),
                  b |= ppl::uniform(0., 2.),
                  r |= ppl::normal(q * w + b, 0.5));

    EXPECT_NEAR(model.pdf(pvalues), 0.055885938549306326, tol);
    EXPECT_NEAR(model.log_pdf(pvalues), -2.884442476988254, tol);
}

}  // namespace ppl
