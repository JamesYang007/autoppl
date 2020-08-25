#include "gtest/gtest.h"
#include <testutil/base_fixture.hpp>
#include <cmath>
#include <array>
#include <autoppl/expression/variable/binary.hpp>
#include <autoppl/expression/variable/constant.hpp>
#include <autoppl/expression/distribution/normal.hpp>
#include <autoppl/expression/op_overloads.hpp>

namespace ppl {

struct normal_integration_fixture: 
    base_fixture<double>,
    ::testing::Test 
{
protected:
    using param_t = Param<value_t, ppl::scl>;
    using data_t = Data<value_t, ppl::vec>;
    using pview_t = typename param_t::base_t;

    std::array<value_t, 2> pvalues = {0.1, -0.1};

    data_t v1;
    param_t x0, x1;

    value_t tol = 1e-15;

    normal_integration_fixture()
        : v1(5)
    {
        v1.get() << 0.1, 0.2, 0.3, 0.4, 0.5;

        // manually set offset
        // in real-use case, user will call an initialization function
        offset_pack_t offset;
        x0.activate(offset);    // updates offset
        x1.activate(offset);

        ptr_pack.uc_val = pvalues.data();
    }
};

TEST_F(normal_integration_fixture, normal_pdfs) {
    
    auto dist1 = normal(0., 1.);
    dist1.bind(ptr_pack);

    EXPECT_NEAR(dist1.pdf(v1), 0.007675723936191419, tol);
    EXPECT_NEAR(dist1.log_pdf(v1), -4.869692666023363, tol);

    auto dist2 = normal(x0, 1.);
    dist2.bind(ptr_pack);

    pvalues[0] = 0.;
    EXPECT_NEAR(dist2.pdf(v1), 0.0076757239361914193, tol);
    EXPECT_NEAR(dist2.log_pdf(v1), -4.869692666023363, tol);

    auto dist3 = normal(x0 + x1, 1.);
    dist3.bind(ptr_pack);

    pvalues[0] = 0.1;
    EXPECT_NEAR(dist3.pdf(v1), 0.0076757239361914193, tol);
    EXPECT_NEAR(dist3.log_pdf(v1), -4.869692666023363, tol);
}

}  // namespace ppl
