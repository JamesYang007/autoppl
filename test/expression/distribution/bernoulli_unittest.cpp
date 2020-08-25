#include "gtest/gtest.h"
#include <fastad>
#include "dist_fixture_base.hpp"
#include <autoppl/expression/distribution/bernoulli.hpp>

namespace ppl {
namespace expr {
namespace dist {

struct bernoulli_fixture : 
    dist_fixture_base<int>,
    dist_fixture_base<double>,
    ::testing::Test
{
protected:
    using disc_base_t = dist_fixture_base<int>;
    using cont_base_t = dist_fixture_base<double>;

    using cont_base_t::infos;
    using cont_base_t::val_buf;
    using cont_base_t::vec_size;

    disc_base_t::value_t x_val_in = 0;
    disc_base_t::value_t x_val_out = -1;
    disc_base_t::vec_t x_vec_in = {0, 1, 1};

    cont_base_t::value_t p_val = 0.6;
    cont_base_t::vec_t p_vec = {0.1, 0.58, 0.99998};

    bernoulli_fixture()
    {
        disc_base_t::val_buf.resize(100); // obscene amount of cache
        val_buf.resize(100); // obscene amount of cache
    }
};

TEST_F(bernoulli_fixture, ctor)
{
    static_assert(util::is_dist_expr_v<Bernoulli<cont_base_t::scl_pv_t>>);
}

TEST_F(bernoulli_fixture, pdf_in)
{
    using bern_t = Bernoulli<cont_base_t::scl_dv_t>;
    disc_base_t::scl_dv_t x(&x_val_in);
    cont_base_t::scl_dv_t p(&p_val);
    bern_t bern(p);
    EXPECT_DOUBLE_EQ(bern.pdf(x), 1-p_val);
}

TEST_F(bernoulli_fixture, log_pdf_in)
{
    using bern_t = Bernoulli<cont_base_t::scl_dv_t>;
    disc_base_t::scl_dv_t x(&x_val_in);
    cont_base_t::scl_dv_t p(&p_val);
    bern_t bern(p);
    EXPECT_DOUBLE_EQ(bern.log_pdf(x), 
                     std::log(1-p_val));
}

TEST_F(bernoulli_fixture, ad_log_pdf)
{
    using bern_t = Bernoulli<cont_base_t::scl_pv_t>;
    disc_base_t::scl_dv_t x(&x_val_in); 
    cont_base_t::scl_pv_t p(&infos[0]);

    bern_t bern(p); 
    bern.bind(cont_base_t::ptr_pack);

    infos[0].off_pack.uc_offset = 0;
    val_buf[infos[0].off_pack.uc_offset] = p_val;

    cont_base_t::ptr_pack.uc_val = val_buf.data();
    auto expr = ad::bind(bern.ad_log_pdf(x, cont_base_t::ptr_pack));
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     std::log(1-p_val));
}

TEST_F(bernoulli_fixture, scl_prune)
{
    using bern_t = Bernoulli<cont_base_t::scl_dv_t>;
    disc_base_t::scl_pv_t x(&infos[0]); 
    cont_base_t::scl_d_t p(0.5);
    bern_t bern(p); 

    disc_base_t::val_buf[0] = -1;
    disc_base_t::ptr_pack.uc_val = disc_base_t::val_buf.data();
    x.bind(disc_base_t::ptr_pack);
    bool pruned = bern.prune(x, x);   // dummy second param
    EXPECT_TRUE(pruned);
    EXPECT_EQ(disc_base_t::val_buf[0], 0);

    disc_base_t::val_buf[0] = 1;
    pruned = bern.prune(x, x);   // dummy second param
    EXPECT_FALSE(pruned);
    EXPECT_EQ(disc_base_t::val_buf[0], 1);
}

TEST_F(bernoulli_fixture, vec_prune)
{
    using bern_t = Bernoulli<cont_base_t::scl_dv_t>;
    disc_base_t::vec_pv_t x(&infos[0], 3); 
    cont_base_t::scl_d_t p(0.5);
    bern_t bern(p); 

    disc_base_t::val_buf[0] = -1;
    disc_base_t::val_buf[1] = 0;
    disc_base_t::val_buf[2] = 1;
    disc_base_t::ptr_pack.uc_val = disc_base_t::val_buf.data();
    x.bind(disc_base_t::ptr_pack);
    bool pruned = bern.prune(x, x);   // dummy second param
    EXPECT_TRUE(pruned);
    EXPECT_EQ(disc_base_t::val_buf[0], 0);
    EXPECT_EQ(disc_base_t::val_buf[1], 0);
    EXPECT_EQ(disc_base_t::val_buf[2], 0);

    disc_base_t::val_buf[0] = 1;
    pruned = bern.prune(x, x);   // dummy second param
    EXPECT_FALSE(pruned);
    EXPECT_EQ(disc_base_t::val_buf[0], 1);
    EXPECT_EQ(disc_base_t::val_buf[1], 0);
    EXPECT_EQ(disc_base_t::val_buf[2], 0);
}

} // namespace dist
} // namespace expr
} // namespace ppl
