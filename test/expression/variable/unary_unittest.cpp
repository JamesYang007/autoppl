#include "gtest/gtest.h"
#include <fastad>
#include <testutil/base_fixture.hpp>
#include <autoppl/expression/variable/unary.hpp>

namespace ppl {
namespace expr {
namespace var {

struct unary_fixture: 
    base_fixture<double>,
    ::testing::Test
{
protected:
    using op_t = ad::core::Exp;
    using scl_unary_t = UnaryNode<op_t, scl_pv_t>;
    using vec_unary_t = UnaryNode<op_t, vec_pv_t>;

    Eigen::VectorXd val_buf;
    
    scl_p_t scl;
    vec_p_t vec;

    scl_unary_t scl_unary;
    vec_unary_t vec_unary;

    unary_fixture()
        : val_buf(4)
        , scl()
        , vec(3)
        , scl_unary(scl)
        , vec_unary(vec)
    {
        // initialize offset of w
        offset_pack_t offset;
        scl.activate(offset);
        vec.activate(offset);

        // initialize values for matrix and pvalues
        val_buf(0) = 1;
        val_buf(1) = -2;
        val_buf(2) = 0;
        val_buf(3) = 0.01;

        ptr_pack.uc_val = val_buf.data();
        scl_unary.bind(ptr_pack);
        vec_unary.bind(ptr_pack);
    }
};

TEST_F(unary_fixture, type_check) 
{
    static_assert(util::is_var_expr_v<scl_unary_t>);
    static_assert(util::is_var_expr_v<vec_unary_t>);
}

TEST_F(unary_fixture, scl_size)
{
    EXPECT_EQ(scl_unary.size(), 1ul);
}

TEST_F(unary_fixture, scl_eval)
{
    value_t res = scl_unary.eval();
    EXPECT_DOUBLE_EQ(res, std::exp(val_buf(0)));
}

TEST_F(unary_fixture, scl_ad)
{
    auto expr = ad::bind(scl_unary.ad(ptr_pack));
    value_t res = ad::evaluate(expr);
    EXPECT_DOUBLE_EQ(res, std::exp(val_buf(0)));
}

TEST_F(unary_fixture, vec_size)
{
    EXPECT_EQ(vec_unary.size(), 3ul);
}

TEST_F(unary_fixture, vec_eval)
{
    Eigen::VectorXd res = vec_unary.eval();
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), std::exp(val_buf(1+i)));
    }
}

TEST_F(unary_fixture, vec_ad)
{
    auto expr = ad::bind(vec_unary.ad(ptr_pack));
    Eigen::VectorXd res = ad::evaluate(expr);
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), std::exp(val_buf(1+i)));
    }
}

} // namespace var
} // namespace expr
} // namespace ppl
