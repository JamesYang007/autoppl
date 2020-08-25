#include "gtest/gtest.h"
#include <testutil/base_fixture.hpp>
#include <cmath>
#include <array>
#include <autoppl/expression/model/bar_eq.hpp>
#include <autoppl/expression/model/glue.hpp>
#include <autoppl/expression/variable/binary.hpp>
#include <autoppl/expression/variable/constant.hpp>
#include <autoppl/expression/distribution/normal.hpp>
#include <autoppl/expression/distribution/uniform.hpp>
#include <autoppl/expression/op_overloads.hpp>
#include <autoppl/util/traits/dist_expr_traits.hpp>

namespace ppl {

struct model_integration_fixture: 
    base_fixture<double>,
    ::testing::Test 
{
protected:
    using data_t = vec_d_t;
    using param_t = scl_p_t;
    using pview_t = typename param_t::base_t;

    value_t tol = 1e-15;

    data_t v1;
    param_t mu, sigma, w, b;
    std::array<value_t, 4> pvalues = {0};

    data_t x;
    data_t y;
    data_t q;
    data_t r;

    model_integration_fixture()
        : v1(5)
        , x(6)
        , y(6)
        , q(6)
        , r(6)
    {
        v1.get() << 0.1, 0.2, 0.3, 0.4, 0.5;
        x.get() << 2.5, 3, 3.5, 4, 4.5, 5.;
        y.get() << 3.5, 4, 4.5, 5, 5.5, 6.;
        q.get() << 2.4, 3.1, 3.6, 4, 4.5, 5.;
        r.get() << 3.5, 4, 4.4, 5.01, 5.46, 6.1;

        // manually set offset
        // in real-use case, user will call an initialization function

        offset_pack_t offset;
        mu.activate(offset);
        sigma.activate(offset);
        w.activate(offset);
        b.activate(offset);

        ptr_pack.uc_val = pvalues.data();
    }
};

TEST_F(model_integration_fixture, simple_model_pdfs) {
    auto model = (
        mu |= uniform(-0.5, 2.),
        v1 |= normal(mu, 1.0)
    );
    model.bind(ptr_pack);

    pvalues[0] = 0.0;

    EXPECT_NEAR(model.pdf(), 0.003070289574476568, tol);
    EXPECT_NEAR(model.log_pdf(), -5.785983397897518, tol);
}

TEST_F(model_integration_fixture, regression_pdfs) {
    pvalues[2] = 1.0;
    pvalues[3] = 1.0;

    auto model = (w |= ppl::uniform(0., 2.),
                  b |= ppl::uniform(0., 2.),
                  r |= ppl::normal(q * w + b, 0.5));

    model.bind(ptr_pack);

    EXPECT_NEAR(model.pdf(), 0.055885938549306326, tol);
    EXPECT_NEAR(model.log_pdf(), -2.884442476988254, tol);
}

}  // namespace ppl
