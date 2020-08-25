#include "gtest/gtest.h"
#include <fastad>
#include <testutil/base_fixture.hpp>
#include <autoppl/expression/variable/dot.hpp>

namespace ppl {
namespace expr {
namespace var {

struct dot_fixture: 
    base_fixture<double>,
    ::testing::Test
{
protected:
    using mv_dot_t = DotNode<mat_dv_t, vec_pv_t>;

    size_t rows = 2;
    size_t cols = 5;

    Eigen::VectorXd pvalues;
    
    mat_d_t mat;
    vec_p_t vec;
    mv_dot_t mv_dot;

    dot_fixture()
        : pvalues(cols)
        , mat(rows, cols)
        , vec(cols)
        , mv_dot(mat, vec)
    {
        // initialize offset of w
        offset_pack_t offset;
        vec.activate(offset);

        // initialize values for matrix and pvalues
        auto& mat_raw = mat.get();
        mat_raw.setZero();
        mat_raw(0,0) = 3.14; mat_raw(0,1) = -0.1; mat_raw(0,3) = 13.24;
        mat_raw(1,0) = -9.2; mat_raw(1,2) = 0.01; mat_raw(1,4) = 6.143;

        pvalues(0) = 1;
        pvalues(1) = -23;
        pvalues(2) = 0;
        pvalues(3) = 0.01;
        pvalues(4) = 0;

        ptr_pack.uc_val = pvalues.data();
        mv_dot.bind(ptr_pack);
    }
};

TEST_F(dot_fixture, type_check) 
{
    static_assert(util::is_var_expr_v<mv_dot_t>);
}

TEST_F(dot_fixture, mv_size)
{
    EXPECT_EQ(mv_dot.size(), rows);
}

TEST_F(dot_fixture, get)
{
    Eigen::MatrixXd actual = mat.get() * pvalues;
    Eigen::MatrixXd res = mv_dot.eval();
    for (size_t i = 0; i < mv_dot.rows(); ++i) {
        for (size_t j = 0; j < mv_dot.cols(); ++j) {
            EXPECT_DOUBLE_EQ(actual(i,j), res(i,j));
        }
    }
}

TEST_F(dot_fixture, to_ad)
{
    auto expr = ad::bind(mv_dot.ad(ptr_pack));
    Eigen::MatrixXd expr_val = ad::evaluate(expr);
    Eigen::MatrixXd actual = mat.get() * pvalues;

    for (size_t i = 0; i < mv_dot.rows(); ++i) {
        for (size_t j = 0; j < mv_dot.cols(); ++j) {
            EXPECT_DOUBLE_EQ(actual(i,j), expr_val(i,j));
        }
    }
}

} // namespace var
} // namespace expr
} // namespace ppl
