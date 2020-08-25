#include "gtest/gtest.h"
#include <fastad>
#include "dist_fixture_base.hpp"
#include <autoppl/expression/distribution/uniform.hpp>

namespace ppl {
namespace expr {
namespace dist {

struct uniform_fixture: 
    dist_fixture_base<double>,
    ::testing::Test 
{
protected:
    // vectors must be size 3 for consistency in this fixture
    value_t x_val_in = 0.;
    value_t x_val_out = -1.;
    vec_t x_vec_in = {0., 0.3, 1.1};
    vec_t x_vec_out = {0., 1., 2.}; // last number is changed to be at the edge of range
    value_t min_val = -1.;
    vec_t min_vec = {-1., 0., 1.};
    value_t max_val = 2.;
    vec_t max_vec = {1., 2., 3.};

    std::mt19937 gen;

    uniform_fixture()
        : gen(0)
    {
        val_buf.resize(100); // obscene amount of cache
    }
};

TEST_F(uniform_fixture, type_check)
{
    using unif_scl_t = Uniform<scl_pv_t, vec_dv_t>;
    static_assert(util::is_dist_expr_v<unif_scl_t>);
}

////////////////////////////////////////////////////////////
// PDF TEST
////////////////////////////////////////////////////////////

TEST_F(uniform_fixture, pdf_in_scl)
{
    using unif_t = Uniform<scl_dv_t, scl_dv_t>;
    scl_dv_t x(&x_val_in);
    scl_dv_t min(&min_val);
    scl_dv_t max(&max_val);
    unif_t unif(min, max);
    EXPECT_DOUBLE_EQ(unif.pdf(x), 1./3);
}

TEST_F(uniform_fixture, pdf_in_vec)
{
    using unif_t = Uniform<vec_dv_t, vec_dv_t>;
    vec_dv_t x(x_vec_in.data(), x_vec_in.size());
    vec_dv_t min(min_vec.data(), min_vec.size());
    vec_dv_t max(max_vec.data(), max_vec.size());
    unif_t unif(min, max);
    EXPECT_DOUBLE_EQ(unif.pdf(x),  0.125);
}

TEST_F(uniform_fixture, pdf_in_scl_vec)
{
    using unif_t = Uniform<scl_dv_t, vec_dv_t>;
    vec_dv_t x(x_vec_in.data(), x_vec_in.size());
    scl_dv_t min(&min_val);
    vec_dv_t max(max_vec.data(), max_vec.size());
    unif_t unif(min, max);
    EXPECT_DOUBLE_EQ(unif.pdf(x), 
                     0.5 * 1./3 * 0.25);
}

TEST_F(uniform_fixture, pdf_out)
{
    using unif_t = Uniform<scl_dv_t, scl_dv_t>;
    scl_dv_t x(&x_val_out);
    scl_dv_t min(&min_val);
    scl_dv_t max(&max_val);
    unif_t unif(min, max);
    EXPECT_DOUBLE_EQ(unif.pdf(x), 0.0);
}

////////////////////////////////////////////////////////////
// Log-PDF TEST
////////////////////////////////////////////////////////////

TEST_F(uniform_fixture, log_pdf_in)
{
    using unif_t = Uniform<scl_dv_t, scl_dv_t>;
    scl_dv_t x(&x_val_in);
    scl_dv_t min(&min_val);
    scl_dv_t max(&max_val);
    unif_t unif(min, max);
    EXPECT_DOUBLE_EQ(unif.log_pdf(x), 
                     -std::log(3.));
}

TEST_F(uniform_fixture, log_pdf_in_scl_vec)
{
    using unif_t = Uniform<scl_dv_t, vec_dv_t>;
    vec_dv_t x(x_vec_in.data(), x_vec_in.size());
    scl_dv_t min(&min_val);
    vec_dv_t max(max_vec.data(), max_vec.size());
    unif_t unif(min, max);
    EXPECT_DOUBLE_EQ(unif.log_pdf(x), 
                     std::log(0.5 * 1./3 * 0.25));
}


TEST_F(uniform_fixture, log_pdf_out)
{
    using unif_t = Uniform<scl_dv_t, scl_dv_t>;
    scl_dv_t x(&x_val_out);
    scl_dv_t min(&min_val);
    scl_dv_t max(&max_val);
    unif_t unif(min, max);
    EXPECT_DOUBLE_EQ(unif.log_pdf(x), 
                     math::neg_inf<value_t>);
}

TEST_F(uniform_fixture, ad_log_pdf) 
{
    using unif_t = Uniform<scl_pv_t, scl_pv_t>;

    scl_pv_t x(&infos[0]);
    scl_pv_t min(&infos[1]);
    scl_pv_t max(&infos[2]);
    unif_t unif(min, max);

    infos[0].off_pack.uc_offset = 0;
    infos[1].off_pack.uc_offset = 1;
    infos[2].off_pack.uc_offset = 2;

    val_buf[0] = x_val_in;
    val_buf[1] = min_val;
    val_buf[2] = max_val;

    ptr_pack.uc_val = val_buf.data();
    auto expr = ad::bind(unif.ad_log_pdf(x, ptr_pack));
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -std::log(max_val - min_val));
}

TEST_F(uniform_fixture, prune_scl)
{
    using unif_t = Uniform<scl_dv_t, scl_dv_t>;
    scl_pv_t x(&infos[0]);
    scl_dv_t min(&min_val);
    scl_dv_t max(&max_val);
    unif_t unif(min, max);

    ptr_pack.uc_val = &x_val_in;
    x.bind(ptr_pack);

    x_val_in = min_val - 1;
    bool pruned = unif.prune(x, gen);
    EXPECT_TRUE(pruned);
    EXPECT_LE(min_val, x_val_in);
    EXPECT_LE(x_val_in, max_val);

    auto in_val = min_val + (max_val - min_val) / 2.;
    x_val_in = in_val;
    pruned = unif.prune(x, gen);
    EXPECT_FALSE(pruned);
    EXPECT_DOUBLE_EQ(x_val_in, in_val);
}

TEST_F(uniform_fixture, prune_vec)
{
    using unif_t = Uniform<vec_dv_t, scl_dv_t>;
    vec_pv_t x(&infos[0], x_vec_in.size());
    vec_dv_t min(min_vec.data(), min_vec.size());
    scl_dv_t max(&max_val);
    unif_t unif(min, max);
    
    ptr_pack.uc_val = x_vec_in.data();
    x.bind(ptr_pack);

    Eigen::Map<Eigen::VectorXd> x_mp(x_vec_in.data(), x_vec_in.size());
    Eigen::Map<Eigen::VectorXd> min_mp(min_vec.data(), min_vec.size());
    x_mp[0] = min_vec[0] - 1;
    bool pruned = unif.prune(x, gen);
    EXPECT_TRUE(pruned);
    bool in = (x_mp.array() >= min_mp.array()).min(
            x_mp.array() <= max_val).all();
    EXPECT_TRUE(in);

    auto in_val = min_val + (max_val - min_val) / 2.;
    x_mp[0] = in_val;
    pruned = unif.prune(x, gen);
    EXPECT_FALSE(pruned);
    EXPECT_DOUBLE_EQ(x_mp[0], in_val);
}

} // namespace dist
} // namespace expr
} // namespace ppl
