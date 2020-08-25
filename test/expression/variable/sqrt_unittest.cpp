#include "gtest/gtest.h"
#include <fastad>
#include <testutil/base_fixture.hpp>
#include <autoppl/expression/variable/sqrt.hpp>

namespace ppl {
namespace expr {
namespace var {

struct sqrt_fixture: 
    base_fixture<double>,
    ::testing::Test
{
protected:
    using scl_sqrt_t = SqrtNode<scl_pv_t>;
    using vec_sqrt_t = SqrtNode<vec_pv_t>;

    Eigen::VectorXd val_buf;
    
    scl_p_t scl;
    vec_p_t vec;

    scl_sqrt_t scl_sqrt;
    vec_sqrt_t vec_sqrt;

    sqrt_fixture()
        : val_buf(4)
        , scl()
        , vec(3)
        , scl_sqrt(scl)
        , vec_sqrt(vec)
    {
        // initialize offset of w
        offset_pack_t offset;
        scl.activate(offset);
        vec.activate(offset);

        // initialize values for matrix and pvalues
        val_buf(0) = 1;
        val_buf(1) = 2;
        val_buf(2) = 0;
        val_buf(3) = 0.01;

        ptr_pack.uc_val = val_buf.data();
        scl_sqrt.bind(ptr_pack);
        vec_sqrt.bind(ptr_pack);
    }
};

TEST_F(sqrt_fixture, type_check) 
{
    static_assert(util::is_var_expr_v<scl_sqrt_t>);
    static_assert(util::is_var_expr_v<vec_sqrt_t>);
}

TEST_F(sqrt_fixture, scl_size)
{
    EXPECT_EQ(scl_sqrt.size(), 1ul);
}

TEST_F(sqrt_fixture, scl_eval)
{
    value_t res = scl_sqrt.eval();
    EXPECT_DOUBLE_EQ(res, std::sqrt(val_buf(0)));
}

TEST_F(sqrt_fixture, scl_ad)
{
    auto expr = ad::bind(scl_sqrt.ad(ptr_pack));
    value_t res = ad::evaluate(expr);
    EXPECT_DOUBLE_EQ(res, std::sqrt(val_buf(0)));
}

TEST_F(sqrt_fixture, vec_size)
{
    EXPECT_EQ(vec_sqrt.size(), 3ul);
}

TEST_F(sqrt_fixture, vec_eval)
{
    Eigen::VectorXd res = vec_sqrt.eval();
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), std::sqrt(val_buf(1+i)));
    }
}

TEST_F(sqrt_fixture, vec_ad)
{
    auto expr = ad::bind(vec_sqrt.ad(ptr_pack));
    Eigen::VectorXd res = ad::evaluate(expr);
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), std::sqrt(val_buf(1+i)));
    }
}

} // namespace var
} // namespace expr
} // namespace ppl
