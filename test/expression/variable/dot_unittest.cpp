#include "gtest/gtest.h"
#include <autoppl/expression/variable/dot.hpp>
#include <autoppl/expression/variable/param.hpp>
#include <autoppl/expression/variable/data.hpp>

namespace ppl {
namespace expr {

struct dot_fixture : ::testing::Test
{
protected:
    using value_t = double;
    using mat_t = arma::Mat<value_t>;
    using p_vec_t = Param<value_t, ppl::vec>;
    using dview_mat_t = DataView<mat_t, ppl::mat>;
    using dot_t = DotNode<dview_mat_t, p_vec_t>;

    static constexpr size_t n_rows = 2;
    static constexpr size_t n_cols = 5;
    
    mat_t mat;
    arma::Col<value_t> pvalues; 
    dview_mat_t X;
    p_vec_t w;
    dot_t dot;

    std::vector<ad::Var<value_t>> ad_vars;
    std::vector<ad::Var<value_t>> ad_cache;

    arma::Col<value_t> actual; // actual matrix product

    dot_fixture()
        : mat(n_rows, n_cols, arma::fill::zeros)
        , pvalues(n_cols, arma::fill::zeros)
        , X(mat)
        , w(n_cols)
        , dot(X, w)
        , ad_vars(n_cols)
        , ad_cache(n_cols)
    {
        // initialize offset of w
        w.set_offset(0);

        // initialize values for matrix and pvalues
        mat(0,0) = 3.14; mat(0,1) = -0.1; mat(0,3) = 13.24;
        mat(1,0) = -9.2; mat(1,2) = 0.01; mat(1,4) = 6.143;

        pvalues(0) = 1;
        pvalues(1) = -23;
        pvalues(3) = 0.01;

        actual = mat * pvalues;

        // initialize ad variables to read values from pvalues
        for (size_t i = 0; i < n_cols; ++i) {
            ad_vars[i].set_value_ptr(&pvalues(i));
        }

        // IMPORTANT: set offset
        dot.set_cache_offset(0);
    }
};

TEST_F(dot_fixture, type_check) 
{
    static_assert(util::is_var_expr_v<dot_t>);
}

TEST_F(dot_fixture, size)
{
    EXPECT_EQ(dot.size(), n_rows);
}

TEST_F(dot_fixture, value)
{
    for (size_t i = 0; i < n_rows; ++i) {
        EXPECT_DOUBLE_EQ(actual(i), dot.value(pvalues, i));
    }
}

TEST_F(dot_fixture, to_ad)
{
    auto expr = dot.to_ad(ad_vars, ad_cache, 0);
    double expr_val = ad::evaluate(expr);
    EXPECT_DOUBLE_EQ(actual(0), expr_val); 

    ad::evaluate_adj(expr);

    // check adjoints
    for (size_t i = 0; i < n_cols; ++i) {
        EXPECT_DOUBLE_EQ(mat(0,i), ad_vars[i].get_adjoint());
    }
}

TEST_F(dot_fixture, to_ad_no_reset_val)
{
    auto expr = dot.to_ad(ad_vars, ad_cache, 0);

    // first eval
    double expr_val = ad::evaluate(expr);
    EXPECT_DOUBLE_EQ(actual(0), expr_val); 

    // second eval should still not affect the cache 
    // even after changing initial values
    pvalues(4) = -0.1232141;
    actual = mat * pvalues;
    expr_val = ad::evaluate(expr);
    EXPECT_DOUBLE_EQ(actual(0), expr_val); 
}

TEST_F(dot_fixture, to_ad_no_reset_adj)
{
    auto expr = dot.to_ad(ad_vars, ad_cache, 0);
    
    // first autodiff
    ad::autodiff(expr);

    // check adjoints
    for (size_t i = 0; i < n_cols; ++i) {
        EXPECT_DOUBLE_EQ(mat(0,i), ad_vars[i].get_adjoint());
    }

    // second autodiff after resetting adjoints
    // of ad vars AND cache variables
    for (auto& v : ad_vars) {
        v.reset_adjoint();
    }
    for (auto& v : ad_cache) {
        v.reset_adjoint();
    }
    pvalues(1) = -13.23;
    pvalues(3) = 0.853;
    ad::autodiff(expr);
    
    // check adjoints
    for (size_t i = 0; i < n_cols; ++i) {
        EXPECT_DOUBLE_EQ(mat(0,i), ad_vars[i].get_adjoint());
    }
}

/*
 * This test shows that multiple expressions built from the 
 * same model and using the same cache is possible, but
 * before backwards evaluating, user has to reset adjoints
 * in cache if cache was used to compute adjoints for an 
 * existing expression.
 */
TEST_F(dot_fixture, to_ad_multiple_exprs_same_cache)
{
    std::vector<ad::Var<value_t>> ad_vars2(n_cols);
    for (size_t i = 0; i < n_cols; ++i) {
        ad_vars2[i].set_value_ptr(&pvalues(i));
    }

    auto expr1 = dot.to_ad(ad_vars, ad_cache, 0);
    auto expr2 = dot.to_ad(ad_vars2, ad_cache, 0);
    
    // first autodiff
    EXPECT_DOUBLE_EQ(ad::autodiff(expr1), actual(0));

    // check adjoints
    for (size_t i = 0; i < n_cols; ++i) {
        EXPECT_DOUBLE_EQ(mat(0,i), ad_vars[i].get_adjoint());
    }

    // second autodiff after resetting adjoints ONLY cache variables
    for (auto& v : ad_cache) {
        v.reset_adjoint();
    }
    pvalues(1) = -13.23;
    pvalues(3) = 0.853;
    actual = mat * pvalues;
    EXPECT_DOUBLE_EQ(ad::autodiff(expr2), actual(0));
    
    // check adjoints
    for (size_t i = 0; i < n_cols; ++i) {
        EXPECT_DOUBLE_EQ(mat(0,i), ad_vars2[i].get_adjoint());
    }
}

/*
 * This test shows that the first expression element
 * must be evaluated before evaluating other elements.
 */
TEST_F(dot_fixture, to_ad_first_elt_eval_first) 
{
    auto expr = dot.to_ad(ad_vars, ad_cache, 1);
    auto res = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(res, 0.);

    // Evaluating expr didn't do anything for v
    // because for_each didn't evaluate anything
    for (const auto& v : ad_vars) {
        EXPECT_DOUBLE_EQ(v.get_adjoint(), 0.);
    }

    // But cache was affected - adjs are updated
    for (size_t i = 0; i < ad_vars.size(); ++i) { 
        EXPECT_DOUBLE_EQ(mat(1,i), ad_cache[i].get_adjoint()); 
    }

    // MUST reset before doing any reverse eval for any expr.
    for (auto& v : ad_cache) { v.reset_adjoint(); }

    auto expr0 = dot.to_ad(ad_vars, ad_cache, 0);
    auto res0 = ad::autodiff(expr0);
    EXPECT_DOUBLE_EQ(actual(0), res0);
    // check that adjoints are updated in this case
    for (size_t i = 0; i < ad_vars.size(); ++i) {
        EXPECT_DOUBLE_EQ(mat(0,i), ad_vars[i].get_adjoint());
    }

    // Now that ad_vars and cache are modified, reset both adjs.
    for (auto& v : ad_vars) { v.reset_adjoint(); }
    for (auto& v : ad_cache) { v.reset_adjoint(); }
   
    // *KEY*: cache currently has fwd evals; can reuse!
    res = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(actual(1), res);
    // check cache adjoints are updated in this case also
    // but NOT ad_vars (0 because we reset) 
    for (size_t i = 0; i < n_cols; ++i) {
        EXPECT_DOUBLE_EQ(0., ad_vars[i].get_adjoint());
        EXPECT_DOUBLE_EQ(mat(1,i), ad_cache[i].get_adjoint());
    }
}


/*
 * Try to differentiate: f = (X*w)[0] + (X*w)[1]
 * Test having 2 expressions built from same dotnode
 * with the same cache, but different ad_vars.
 */
TEST_F(dot_fixture, sum_first_two_comp) 
{
    auto expr = 
        dot.to_ad(ad_vars, ad_cache, 0) +
        dot.to_ad(ad_vars, ad_cache, 1);

    std::vector<ad::Var<value_t>> ad_vars2(ad_vars.size());
    for (size_t i = 0; i < ad_vars2.size(); ++i) {
        ad_vars2[i].set_value_ptr(&pvalues[i]);
    }

    auto expr2 = 
        dot.to_ad(ad_vars2, ad_cache, 0) +
        dot.to_ad(ad_vars2, ad_cache, 1);

    // first expr autodiff
    auto res = ad::autodiff(expr);
    EXPECT_DOUBLE_EQ(res, actual(0) + actual(1));
    for (size_t i = 0; i < ad_vars.size(); ++i) {
        EXPECT_DOUBLE_EQ(mat(0,i) + mat(1,i), 
                         ad_vars[i].get_adjoint());
        EXPECT_DOUBLE_EQ(0., 
                         ad_vars2[i].get_adjoint());
    }

    // must renew cache
    for (auto& v : ad_cache) { v.reset_adjoint(); }

    // second expr autodiff
    // first ad_var adjoints should have remained the same
    res = ad::autodiff(expr2);
    EXPECT_DOUBLE_EQ(res, actual(0) + actual(1));
    for (size_t i = 0; i < ad_vars.size(); ++i) {
        EXPECT_DOUBLE_EQ(mat(0,i) + mat(1,i), 
                         ad_vars[i].get_adjoint());
        EXPECT_DOUBLE_EQ(mat(0,i) + mat(1,i), 
                         ad_vars2[i].get_adjoint());
    }
}

} // namespace expr
} // namespace ppl
