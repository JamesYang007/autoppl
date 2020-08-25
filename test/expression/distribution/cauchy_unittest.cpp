#include "gtest/gtest.h"
#include <fastad>
#include "dist_fixture_base.hpp"
#include <autoppl/expression/distribution/cauchy.hpp>

namespace ppl {
namespace expr {
namespace dist {

struct cauchy_fixture: 
    dist_fixture_base<double>,
    ::testing::Test 
{
protected:
    // vectors must be size 3 for consistency in this fixture
    value_t x_val = 0.421;
    vec_t x_vec = {0.5, -1.3, -3.2414999};
    value_t loc_val = 0.341;
    vec_t loc_vec = {0.4, -2.30000001, -10.32};
    value_t scale_val = 2.132;
    vec_t scale_vec = {0.51, 0.01, 3.4};

    std::mt19937 gen;

    cauchy_fixture()
        : gen(0)
    {
        val_buf.resize(100); // obscene amount of cache
    }
};

TEST_F(cauchy_fixture, type_check)
{
    using cauchy_scl_t = Cauchy<scl_pv_t, vec_dv_t>;
    static_assert(util::is_dist_expr_v<cauchy_scl_t>);
}

////////////////////////////////////////////////////////////
// Log-PDF TEST
////////////////////////////////////////////////////////////

TEST_F(cauchy_fixture, log_pdf)
{
    using cauchy_t = Cauchy<scl_dv_t, scl_dv_t>;
    scl_dv_t x(&x_val);
    scl_dv_t loc(&loc_val);
    scl_dv_t scale(&scale_val);
    cauchy_t cauchy(loc, scale);
    EXPECT_DOUBLE_EQ(cauchy.log_pdf(x), 
                     -0.7584675254495730);
}

TEST_F(cauchy_fixture, log_pdf_scl_vec)
{
    using cauchy_t = Cauchy<scl_dv_t, vec_dv_t>;
    vec_dv_t x(x_vec.data(), x_vec.size());
    scl_dv_t loc(&loc_val);
    vec_dv_t scale(scale_vec.data(), scale_vec.size());
    cauchy_t cauchy(loc, scale);
    EXPECT_DOUBLE_EQ(cauchy.log_pdf(x), 
                     -6.9858076420069022);
}

TEST_F(cauchy_fixture, ad_log_pdf) 
{
    using cauchy_t = Cauchy<scl_pv_t, scl_pv_t>;

    scl_pv_t x(&infos[0]);
    scl_pv_t loc(&infos[1]);
    scl_pv_t scale(&infos[2]);
    cauchy_t cauchy(loc, scale);

    infos[0].off_pack.uc_offset = 0;
    infos[1].off_pack.uc_offset = 1;
    infos[2].off_pack.uc_offset = 2;

    val_buf[0] = x_val;
    val_buf[1] = loc_val;
    val_buf[2] = scale_val;

    ptr_pack.uc_val = val_buf.data();
    auto expr = ad::bind(cauchy.ad_log_pdf(x, ptr_pack));
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -0.7584675254495730);
}

} // namespace dist
} // namespace expr
} // namespace ppl
