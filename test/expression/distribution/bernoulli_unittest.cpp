#include "gtest/gtest.h"
#include "dist_fixture_base.hpp"
#include <autoppl/expression/distribution/bernoulli.hpp>
#include <autoppl/util/traits/mock_types.hpp>

namespace ppl {
namespace expr {

struct bernoulli_fixture : 
    dist_fixture_base<int>,
    dist_fixture_base<double>,
    ::testing::Test
{
protected:
    using disc_base_t = dist_fixture_base<int>;
    using cont_base_t = dist_fixture_base<double>;

    using cont_base_t::offsets;
    using cont_base_t::storage;
    using cont_base_t::cache;
    using cont_base_t::vec_size;

    disc_base_t::value_t x_val_in = 0;
    disc_base_t::value_t x_val_out = -1;
    disc_base_t::vec_t x_vec_in = {0, 1, 1};

    cont_base_t::value_t p_val = 0.6;
    cont_base_t::vec_t p_vec = {0.1, 0.58, 0.99998};

    bernoulli_fixture()
    {
        cache.resize(100); // obscene amount of cache
    }
};

TEST_F(bernoulli_fixture, ctor)
{
    static_assert(util::is_dist_expr_v<Bernoulli<MockVarExpr>>);
}

TEST_F(bernoulli_fixture, pdf_in)
{
    using bern_t = Bernoulli<cont_base_t::dv_scl_t>;
    disc_base_t::dv_scl_t x(x_val_in);
    cont_base_t::dv_scl_t p(p_val);
    bern_t bern(p);
    cont_base_t::vec_t pvalues;  // no parameter values
    EXPECT_DOUBLE_EQ(bern.pdf(x, pvalues), 
                     1-p_val);
}

TEST_F(bernoulli_fixture, pdf_out)
{
    using bern_t = Bernoulli<cont_base_t::dv_scl_t>;
    disc_base_t::dv_scl_t x(x_val_out);
    cont_base_t::dv_scl_t p(p_val);
    bern_t bern(p);
    cont_base_t::vec_t pvalues;  // no parameter values
    EXPECT_DOUBLE_EQ(bern.pdf(x, pvalues), 
                     0.);
}

TEST_F(bernoulli_fixture, log_pdf_in)
{
    using bern_t = Bernoulli<cont_base_t::dv_scl_t>;
    disc_base_t::dv_scl_t x(x_val_in);
    cont_base_t::dv_scl_t p(p_val);
    bern_t bern(p);
    cont_base_t::vec_t pvalues;  // no parameter values
    EXPECT_DOUBLE_EQ(bern.log_pdf(x, pvalues), 
                     std::log(1-p_val));
}

TEST_F(bernoulli_fixture, log_pdf_out)
{
    using bern_t = Bernoulli<cont_base_t::dv_scl_t>;
    disc_base_t::dv_scl_t x(x_val_out);
    cont_base_t::dv_scl_t p(p_val);
    bern_t bern(p);
    cont_base_t::vec_t pvalues;  // no parameter values
    EXPECT_DOUBLE_EQ(bern.log_pdf(x, pvalues), 
                     math::neg_inf<cont_base_t::value_t>);
}

/////////////////////////////////////////////////////////////////
// TEST ad_log_pdf
/////////////////////////////////////////////////////////////////

// Case 1
TEST_F(bernoulli_fixture, ad_log_pdf_case1_in)
{
    using bern_t = Bernoulli<cont_base_t::pv_scl_t>;
    disc_base_t::dv_scl_t x(x_val_in); 
    cont_base_t::pv_scl_t p(offsets[0], storage[0]);

    bern_t bern(p); 
    bern.set_cache_offset(0);

    offsets[0] = 0;

    cont_base_t::ad_vec_t ad_vars(1);
    ad_vars[0].set_value(p_val);

    auto expr = bern.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     std::log(1-p_val));
}

TEST_F(bernoulli_fixture, ad_log_pdf_case1_out)
{
    using bern_t = Bernoulli<cont_base_t::pv_scl_t>;
    disc_base_t::dv_scl_t x(x_val_out); 
    cont_base_t::pv_scl_t p(offsets[0], storage[0]);

    bern_t bern(p); 
    bern.set_cache_offset(0);

    offsets[0] = 0;

    cont_base_t::ad_vec_t ad_vars(1);
    ad_vars[0].set_value(p_val);

    auto expr = bern.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     math::neg_inf<typename cont_base_t::value_t>);
}

// Case 2: undefined behavior if x is not in the range
TEST_F(bernoulli_fixture, ad_log_pdf_case2)
{
    using bern_t = Bernoulli<cont_base_t::pv_scl_t>;
    disc_base_t::dv_vec_t x(x_vec_in); 
    cont_base_t::pv_scl_t p(offsets[0], storage[0]);

    bern_t bern(p); 
    bern.set_cache_offset(0);

    offsets[0] = 0;

    cont_base_t::ad_vec_t ad_vars(1);
    ad_vars[0].set_value(p_val);

    auto expr = bern.ad_log_pdf(x, ad_vars, cache);
    
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     2*std::log(p_val) + std::log(1-p_val));
}

// Case 3: undefined behavior if x is not in the range
TEST_F(bernoulli_fixture, ad_log_pdf_case3)
{
    using bern_t = Bernoulli<cont_base_t::pv_vec_t>;
    disc_base_t::dv_vec_t x(x_vec_in); 
    cont_base_t::pv_vec_t p(offsets[0], storage, vec_size);

    bern_t bern(p); 
    bern.set_cache_offset(0);

    offsets[0] = 0;

    cont_base_t::ad_vec_t ad_vars(p.size());
    for (size_t i = 0; i < ad_vars.size(); ++i) {
        ad_vars[i].set_value(p_vec[i]);
    } 

    auto expr = bern.ad_log_pdf(x, ad_vars, cache);

    double actual = 0;
    for (size_t i = 0; i < p.size(); ++i) { 
        if (x_vec_in[i] == 1) actual += std::log(p_vec[i]);
        else actual += std::log(1-p_vec[i]);
    }
    
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     actual);
}

} // namespace expr
} // namespace ppl
