#include "gtest/gtest.h"
#include "dist_fixture_base.hpp"
#include <autoppl/expression/distribution/normal.hpp>
#include <autoppl/util/traits/mock_types.hpp>

namespace ppl {
namespace expr {

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
        this->cache.resize(100);  // obscene amount of cache
    }
};

TEST_F(normal_fixture, type_check)
{
    using norm_scl_t = Normal<MockVarExpr, MockVarExpr>;
    static_assert(util::is_dist_expr_v<norm_scl_t>);
}

TEST_F(normal_fixture, pdf)
{
    using norm_t = Normal<dv_scl_t, dv_scl_t>;
    dv_vec_t x(x_vec);
    dv_scl_t mean(mean_val);
    dv_scl_t sd(sd_val);
    norm_t norm(mean, sd);
    vec_t pvalues;  // no parameter values
    EXPECT_DOUBLE_EQ(norm.pdf(x, pvalues), 
                     0.005211875018288502);
}

TEST_F(normal_fixture, log_pdf)
{
    using norm_t = Normal<dv_scl_t, dv_scl_t>;
    dv_vec_t x(x_vec);
    dv_scl_t mean(mean_val);
    dv_scl_t sd(sd_val);
    norm_t norm(mean, sd);
    vec_t pvalues; // no parameter values
    EXPECT_DOUBLE_EQ(norm.log_pdf(x, pvalues), 
                     -5.2568155996140185);
}

// AD log pdf case 1, subcase 1
TEST_F(normal_fixture, ad_log_pdf_case_11)
{
    using norm_t = Normal<dv_scl_t, dv_scl_t>;
    dv_scl_t x(x_val);
    dv_scl_t mean(mean_val);
    dv_scl_t sd(sd_val);
    norm_t norm(mean, sd);

    auto expr = norm.ad_log_pdf(x, cache, cache);  // last two param unused

    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -0.020000000000000018);
}

// AD log pdf case 1, subcase 2 when x has param
TEST_F(normal_fixture, ad_log_pdf_case_12_xparam)
{
    using norm_t = Normal<dv_scl_t, pv_scl_t>;

    ad_vec_t ad_vars(2);
    ad_vars[0].set_value(x_val);
    ad_vars[1].set_value(sd_val);

    // initialize offsets that params will view
    // MUST correspond to begin indices in ad_vars
    offsets[0] = 0;
    offsets[1] = 1;

    pv_scl_t x(offsets[0], storage[0]);    // storage not used 
    dv_scl_t mean(mean_val);
    pv_scl_t sd(offsets[1], storage[1]); // storage not used
    norm_t norm(mean, sd);
    norm.set_cache_offset(0);

    auto expr = norm.ad_log_pdf(x, ad_vars, cache);

    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -0.020000000000000018);
}

// AD log pdf case 1, subcase 2 when mean has param
TEST_F(normal_fixture, ad_log_pdf_case_12_mparam)
{
    using norm_t = Normal<pv_scl_t, pv_scl_t>;

    ad_vec_t ad_vars(2);
    ad_vars[0].set_value(mean_val);
    ad_vars[1].set_value(sd_val);

    offsets[0] = 0;
    offsets[1] = 1;

    dv_scl_t x(x_val);
    pv_scl_t mean(offsets[0], storage[0]);
    pv_scl_t sd(offsets[1], storage[1]);
    norm_t norm(mean, sd);
    norm.set_cache_offset(0);

    auto expr = norm.ad_log_pdf(x, ad_vars, cache);

    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -0.020000000000000018);
}

// AD log pdf case 1, subcase 3
TEST_F(normal_fixture, ad_log_pdf_case_13)
{
    using norm_t = Normal<dv_scl_t, pv_scl_t>;

    ad_vec_t ad_vars(1);
    ad_vars[0].set_value(sd_val);

    offsets[0] = 0;

    dv_scl_t x(x_val);
    dv_scl_t mean(mean_val);
    pv_scl_t sd(offsets[0], storage[0]);
    norm_t norm(mean, sd);
    norm.set_cache_offset(0);

    auto expr = norm.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -0.020000000000000018);
}

// AD log pdf case 2, subcase 1
TEST_F(normal_fixture, ad_log_pdf_case_21)
{
    using norm_t = Normal<dv_scl_t, dv_scl_t>;

    offsets[0] = 0;

    pv_vec_t x(offsets[0], storage, vec_size);
    dv_scl_t mean(mean_val);
    dv_scl_t sd(sd_val);
    norm_t norm(mean, sd);
    norm.set_cache_offset(0);

    ad_vec_t ad_vars(x_vec.size());
    std::for_each(util::counting_iterator<size_t>(0),
                  util::counting_iterator<size_t>(x_vec.size()),
                  [&](size_t i) { ad_vars[i].set_value(x_vec[i]); });

    auto expr = norm.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -2.5000000000000004);
}

// AD log pdf case 2, subcase 2
TEST_F(normal_fixture, ad_log_pdf_case_22)
{
    using norm_t = Normal<pv_scl_t, pv_scl_t>;

    ad_vec_t ad_vars(2);
    ad_vars[0].set_value(mean_val);
    ad_vars[1].set_value(sd_val);

    offsets[0] = 0;
    offsets[1] = 1;

    dv_vec_t x(x_vec);
    pv_scl_t mean(offsets[0], storage[0]);
    pv_scl_t sd(offsets[1], storage[1]);
    norm_t norm(mean, sd);
    norm.set_cache_offset(0);

    auto expr = norm.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -2.5000000000000004);
}

// AD log pdf case 3
TEST_F(normal_fixture, ad_log_pdf_case_3)
{
    using norm_t = Normal<pv_vec_t, pv_scl_t>;

    ad_vec_t ad_vars(vec_size + 1);

    std::for_each(util::counting_iterator<>(0),
                  util::counting_iterator<>(vec_size),
                  [&](auto i) { ad_vars[i].set_value(mean_vec[i]); });
    ad_vars[vec_size].set_value(sd_val);

    offsets[0] = 0;
    offsets[1] = offsets[0] + vec_size;

    dv_vec_t x(x_vec);
    pv_vec_t mean(offsets[0], storage, vec_size);
    pv_scl_t sd(offsets[1], storage[vec_size]);
    norm_t norm(mean, sd);
    norm.set_cache_offset(0);

    auto expr = norm.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -1.5000000000000004);
}

// AD log pdf case 4, subcase 1 
TEST_F(normal_fixture, ad_log_pdf_case_41)
{
    using norm_t = Normal<pv_scl_t, pv_vec_t>;

    ad_vec_t ad_vars(vec_size + 1);

    ad_vars[0].set_value(mean_val);
    std::for_each(util::counting_iterator<>(0),
                  util::counting_iterator<>(vec_size),
                  [&](auto i) { ad_vars[i+1].set_value(sd_vec[i]); });

    offsets[0] = 0;
    offsets[1] = 1; 

    dv_vec_t x(x_vec);
    pv_scl_t mean(offsets[0], storage[0]);
    pv_vec_t sd(offsets[1], storage, vec_size);
    norm_t norm(mean, sd);
    norm.set_cache_offset(0);

    auto expr = norm.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -2.1389816914502773);
}

// AD log pdf case 4, subcase 2
TEST_F(normal_fixture, ad_log_pdf_case_42)
{
    using norm_t = Normal<pv_scl_t, dv_vec_t>;

    ad_vec_t ad_vars(vec_size + 1);

    std::for_each(util::counting_iterator<>(0),
                  util::counting_iterator<>(vec_size),
                  [&](auto i) { ad_vars[i].set_value(x_vec[i]); });
    ad_vars[vec_size].set_value(mean_val);

    offsets[0] = 0;
    offsets[1] = vec_size + offsets[0]; 

    pv_vec_t x(offsets[0], storage, vec_size);
    pv_scl_t mean(offsets[1], storage[0]);
    dv_vec_t sd(sd_vec);
    norm_t norm(mean, sd);
    norm.set_cache_offset(0);

    auto expr = norm.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -2.1389816914502773);
}

// AD log pdf case 4, subcase 3
TEST_F(normal_fixture, ad_log_pdf_case_43)
{
    using norm_t = Normal<pv_scl_t, dv_vec_t>;

    ad_vec_t ad_vars(1);
    ad_vars[0].set_value(mean_val);

    offsets[0] = 0;

    dv_vec_t x(x_vec);
    pv_scl_t mean(offsets[0], storage[0]);
    dv_vec_t sd(sd_vec);
    norm_t norm(mean, sd);
    norm.set_cache_offset(0);

    auto expr = norm.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -2.1389816914502773);
}

// AD log pdf case 5
TEST_F(normal_fixture, ad_log_pdf_case_5)
{
    using norm_t = Normal<pv_vec_t, pv_vec_t>;

    ad_vec_t ad_vars(2*vec_size);

    for (size_t i = 0; i < vec_size; ++i) {
        ad_vars[i].set_value(mean_vec[i]);
        ad_vars[i+vec_size].set_value(sd_vec[i]);
    }

    offsets[0] = 0;
    offsets[1] = vec_size;

    dv_vec_t x(x_vec);
    pv_vec_t mean(offsets[0], storage, vec_size);
    pv_vec_t sd(offsets[1], storage, vec_size);
    norm_t norm(mean, sd);
    norm.set_cache_offset(0);

    auto expr = norm.ad_log_pdf(x, ad_vars, cache);
    EXPECT_DOUBLE_EQ(ad::evaluate(expr),
                     -2.4723150247836103);
}

} // namespace expr
} // namespace ppl
