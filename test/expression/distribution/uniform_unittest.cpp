#include "gtest/gtest.h"
#include "dist_fixture_base.hpp"
#include <autoppl/expression/distribution/uniform.hpp>
#include <autoppl/util/traits/mock_types.hpp>

namespace ppl {
namespace expr {

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

    uniform_fixture()
    {
        this->cache.resize(100); // obscene amount of cache
    }
};

TEST_F(uniform_fixture, type_check)
{
    using unif_scl_t = Uniform<MockVarExpr, MockVarExpr>;
    static_assert(util::is_dist_expr_v<unif_scl_t>);
}

////////////////////////////////////////////////////////////
// PDF TEST
////////////////////////////////////////////////////////////

TEST_F(uniform_fixture, pdf_in_scl)
{
    using unif_t = Uniform<dv_scl_t, dv_scl_t>;
    dv_scl_t x(x_val_in);
    dv_scl_t min(min_val);
    dv_scl_t max(max_val);
    unif_t unif(min, max);
    vec_t pvalues;  // no parameter values
    EXPECT_DOUBLE_EQ(unif.pdf(x, pvalues), 
                     1./3);
}

TEST_F(uniform_fixture, pdf_in_vec)
{
    using unif_t = Uniform<dv_vec_t, dv_vec_t>;
    dv_vec_t x(x_vec_in);
    dv_vec_t min(min_vec);
    dv_vec_t max(max_vec);
    unif_t unif(min, max);
    vec_t pvalues;  // no parameter values
    EXPECT_DOUBLE_EQ(unif.pdf(x, pvalues), 
                     0.125);
}

TEST_F(uniform_fixture, pdf_in_scl_vec)
{
    using unif_t = Uniform<dv_scl_t, dv_vec_t>;
    dv_vec_t x(x_vec_in);
    dv_scl_t min(min_val);
    dv_vec_t max(max_vec);
    unif_t unif(min, max);
    vec_t pvalues;  // no parameter values
    EXPECT_DOUBLE_EQ(unif.pdf(x, pvalues), 
                     0.5 * 1./3 * 0.25);
}

TEST_F(uniform_fixture, pdf_out)
{
    using unif_t = Uniform<dv_scl_t, dv_scl_t>;
    dv_scl_t x(x_val_out);
    dv_scl_t min(min_val);
    dv_scl_t max(max_val);
    unif_t unif(min, max);
    vec_t pvalues;  // no parameter values
    EXPECT_DOUBLE_EQ(unif.pdf(x, pvalues), 
                     0.0);
}

////////////////////////////////////////////////////////////
// Log-PDF TEST
////////////////////////////////////////////////////////////

TEST_F(uniform_fixture, log_pdf_in)
{
    using unif_t = Uniform<dv_scl_t, dv_scl_t>;
    dv_scl_t x(x_val_in);
    dv_scl_t min(min_val);
    dv_scl_t max(max_val);
    unif_t unif(min, max);
    vec_t pvalues;  // no parameter values
    EXPECT_DOUBLE_EQ(unif.log_pdf(x, pvalues), 
                     -std::log(3.));
}

TEST_F(uniform_fixture, log_pdf_in_scl_vec)
{
    using unif_t = Uniform<dv_scl_t, dv_vec_t>;
    dv_vec_t x(x_vec_in);
    dv_scl_t min(min_val);
    dv_vec_t max(max_vec);
    unif_t unif(min, max);
    vec_t pvalues;  // no parameter values
    EXPECT_DOUBLE_EQ(unif.log_pdf(x, pvalues), 
                     std::log(0.5 * 1./3 * 0.25));
}


TEST_F(uniform_fixture, log_pdf_out)
{
    using unif_t = Uniform<dv_scl_t, dv_scl_t>;
    dv_scl_t x(x_val_out);
    dv_scl_t min(min_val);
    dv_scl_t max(max_val);
    unif_t unif(min, max);
    vec_t pvalues;  // no parameter values
    EXPECT_DOUBLE_EQ(unif.log_pdf(x, pvalues), 
                     math::neg_inf<value_t>);
}

////////////////////////////////////////////////////////////
// ad_log_pdf TEST
////////////////////////////////////////////////////////////

// Case 1:
TEST_F(uniform_fixture, ad_log_pdf_case1) 
{
    using unif_t = Uniform<pv_scl_t, pv_scl_t>;

    // storage is ignored for now
    pv_scl_t x(offsets[0], storage[0]);
    pv_scl_t min(offsets[1], storage[1]);
    pv_scl_t max(offsets[2], storage[2]);
    unif_t unif(min, max);

    unif.set_cache_offset(0);

    offsets[0] = 0;
    offsets[1] = 1;
    offsets[2] = 2;

    ad_vec_t ad_vars(3);
    ad_vars[0].set_value(x_val_in);
    ad_vars[1].set_value(min_val);
    ad_vars[2].set_value(max_val);

    auto expr = unif.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -std::log(max_val - min_val));
}

// Case 2, Subcase 1:
TEST_F(uniform_fixture, ad_log_pdf_case21) 
{
    using unif_t = Uniform<dv_scl_t, dv_scl_t>;
    dv_vec_t x(x_vec_in);
    dv_scl_t min(min_val);
    dv_scl_t max(max_val);
    unif_t unif(min, max);

    unif.set_cache_offset(0);

    ad_vec_t ad_vars;

    auto expr = unif.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -std::log(27.));
}

// Case 2, Subcase 2:
TEST_F(uniform_fixture, ad_log_pdf_case22) 
{
    using unif_t = Uniform<dv_scl_t, dv_scl_t>;
    pv_vec_t x(offsets[0], storage, vec_size);
    dv_scl_t min(min_val);
    dv_scl_t max(max_val);
    unif_t unif(min, max);

    unif.set_cache_offset(0);

    offsets[0] = 0;

    ad_vec_t ad_vars(vec_size);
    std::for_each(util::counting_iterator<>(0),
                  util::counting_iterator<>(vec_size),
                  [&](size_t i) { ad_vars[i].set_value(x_vec_in[i]); });

    auto expr = unif.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -std::log(27.));
}

// Case 3:
TEST_F(uniform_fixture, ad_log_pdf_case3) 
{
    using unif_t = Uniform<pv_vec_t, pv_scl_t>;

    // storage is ignored for now
    pv_vec_t x(offsets[0], storage, vec_size);
    pv_vec_t min(offsets[1], storage, vec_size);
    pv_scl_t max(offsets[2], storage[0]);
    unif_t unif(min, max);

    unif.set_cache_offset(0);

    offsets[0] = 0;
    offsets[1] = vec_size;
    offsets[2] = 2*vec_size;

    ad_vec_t ad_vars(vec_size * 2 + 1);
    ad_vars[2*vec_size].set_value(max_val);

    std::for_each(util::counting_iterator<>(0),
                  util::counting_iterator<>(vec_size),
                  [&](size_t i) { 
                    ad_vars[i].set_value(x_vec_in[i]); 
                    ad_vars[i+vec_size].set_value(min_vec[i]); 
                });

    double actual = 0;
    for (auto m : min_vec) {
        actual -= std::log(max_val - m);
    }

    auto expr = unif.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     actual);
}

// Case 4:
TEST_F(uniform_fixture, ad_log_pdf_case4) 
{
    using unif_t = Uniform<dv_scl_t, pv_vec_t>;

    // storage is ignored for now
    pv_vec_t x(offsets[0], storage, vec_size);
    dv_scl_t min(min_val);
    pv_vec_t max(offsets[1], storage, vec_size);
    unif_t unif(min, max);

    unif.set_cache_offset(0);

    offsets[0] = 0;
    offsets[1] = vec_size;

    ad_vec_t ad_vars(vec_size * 2);
    std::for_each(util::counting_iterator<>(0),
                  util::counting_iterator<>(vec_size),
                  [&](size_t i) { 
                    ad_vars[i].set_value(x_vec_in[i]); 
                    ad_vars[i+vec_size].set_value(max_vec[i]); 
                });

    double actual = 0;
    for (auto m : max_vec) {
        actual -= std::log(m - min_val);
    }

    auto expr = unif.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     actual);
}

// Case 5:
TEST_F(uniform_fixture, ad_log_pdf_case5)
{
    using unif_t = Uniform<pv_vec_t, pv_vec_t>;

    // storage is ignored for now
    pv_vec_t x(offsets[0], storage, vec_size);
    pv_vec_t min(offsets[1], storage, vec_size);
    pv_vec_t max(offsets[2], storage, vec_size);
    unif_t unif(min, max);

    unif.set_cache_offset(0);

    offsets[0] = 0;
    offsets[1] = vec_size;
    offsets[2] = vec_size * 2;

    ad_vec_t ad_vars(vec_size * 3);
    std::for_each(util::counting_iterator<>(0),
                  util::counting_iterator<>(vec_size),
                  [&](size_t i) { 
                    ad_vars[i].set_value(x_vec_in[i]); 
                    ad_vars[i+vec_size].set_value(min_vec[i]); 
                    ad_vars[i+2*vec_size].set_value(max_vec[i]); 
                });

    double actual = 0;
    for (size_t i = 0; i < min_vec.size(); ++i) {
        actual -= std::log(max_vec[i] - min_vec[i]);
    }

    auto expr = unif.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     actual);
}

} // namespace expr
} // namespace ppl
