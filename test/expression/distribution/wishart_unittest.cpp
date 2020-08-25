#include "gtest/gtest.h"
#include <fastad>
#include "dist_fixture_base.hpp"
#include <autoppl/expression/distribution/wishart.hpp>

namespace ppl {
namespace expr {
namespace dist {

struct wishart_fixture: 
    dist_fixture_base<double>,
    ::testing::Test
{
protected:
    mat_t x_val;
    mat_t v_val;
    value_t n_val = 3.;

    wishart_fixture()
        : x_val({2., 1., 1., 2.})
        , v_val({1., 0., 0., 1.})
    {
        val_buf.resize(100);  // obscene amount of cache
    }
};

TEST_F(wishart_fixture, type_check)
{
    using wishart_t = Wishart<mat_pv_t, scl_dv_t>;
    static_assert(util::is_dist_expr_v<wishart_t>);
}

TEST_F(wishart_fixture, log_pdf)
{
    using wishart_t = Wishart<mat_dv_t, scl_dv_t>;
    mat_dv_t x(x_val.data(), 2, 2);
    mat_dv_t v(v_val.data(), 2, 2);
    scl_dv_t n(&n_val);
    wishart_t wishart(v, n);
    EXPECT_DOUBLE_EQ(wishart.log_pdf(x), -2.);
}

TEST_F(wishart_fixture, prune)
{
    using wishart_t = Wishart<mat_dv_t, scl_dv_t>;
    mat_dv_t x(x_val.data(), 2, 2);
    mat_dv_t v(v_val.data(), 2, 2);
    scl_dv_t n(&n_val);
    wishart_t wishart(v, n);
    bool pruned = wishart.prune(x,x); // dummy params
    EXPECT_FALSE(pruned);
}

} // namespace dist
} // namespace expr
} // namespace ppl
