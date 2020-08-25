#include "gtest/gtest.h"
#include <fastad>
#include "dist_fixture_base.hpp"
#include <autoppl/expression/distribution/normal.hpp>

namespace ppl {
namespace expr {
namespace dist {

struct normal_fixture: 
    dist_fixture_base<double>,
    ::testing::Test
{
protected:
    // vectors must be size 3 for consistency in this fixture
    value_t x_val = -0.2;
    vec_t x_vec = {0., 1., 2.};
    value_t mean_val = 0.;
    vec_t mean_vec = {-1., 0., 1.};
    value_t sd_val = 1.;
    vec_t sd_vec = {1., 2., 3.};

    normal_fixture()
    {
        val_buf.resize(100);  // obscene amount of cache
    }
};

TEST_F(normal_fixture, type_check)
{
    using norm_scl_t = Normal<scl_pv_t, mat_pv_t>;
    static_assert(util::is_dist_expr_v<norm_scl_t>);
}

TEST_F(normal_fixture, pdf)
{
    using norm_t = Normal<scl_dv_t, scl_dv_t>;
    vec_dv_t x(x_vec.data(), x_vec.size());
    scl_dv_t mean(&mean_val);
    scl_dv_t sd(&sd_val);
    norm_t norm(mean, sd);
    EXPECT_DOUBLE_EQ(norm.pdf(x), 
                     0.005211875018288502);
}

TEST_F(normal_fixture, log_pdf)
{
    using norm_t = Normal<scl_dv_t, scl_dv_t>;
    vec_dv_t x(x_vec.data(), x_vec.size());
    scl_dv_t mean(&mean_val);
    scl_dv_t sd(&sd_val);
    norm_t norm(mean, sd);
    EXPECT_DOUBLE_EQ(norm.log_pdf(x), 
                     -5.2568155996140185);
}

TEST_F(normal_fixture, ad_log_pdf)
{
    using norm_t = Normal<scl_dv_t, scl_dv_t>;
    scl_dv_t x(&x_val);
    scl_dv_t mean(&mean_val);
    scl_dv_t sd(&sd_val);
    norm_t norm(mean, sd);

    auto expr = ad::bind(norm.ad_log_pdf(x, ptr_pack));

    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -0.020000000000000018);
}

TEST_F(normal_fixture, prune)
{
    using norm_t = Normal<scl_dv_t, scl_dv_t>;
    scl_dv_t mean(&mean_val);
    scl_dv_t sd(&sd_val);
    norm_t norm(mean, sd);
    bool pruned = norm.prune(mean, mean); // dummy params
    EXPECT_FALSE(pruned);
}

} // namespace dist
} // namespace expr
} // namespace ppl
