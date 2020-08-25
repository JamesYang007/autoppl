#include <gtest/gtest.h>
#include <testutil/base_fixture.hpp>
#include <autoppl/expression/constraint/lower.hpp>

namespace ppl {
namespace expr {
namespace constraint {

struct lower_fixture:
    base_fixture<double>,
    ::testing::Test
{
protected:
    using lower_t = Lower<scl_dv_t>;
    value_t tol = 1e-15;
    scl_d_t expr;
    lower_t constraint;

    Eigen::VectorXd val_buf;

    lower_fixture()
        : expr(0.32)
        , constraint(expr)
        , val_buf(1)
    {
        ptr_pack.uc_val = val_buf.data();
        constraint.activate_refcnt();
        constraint.bind(ptr_pack);
    }

    value_t transform(value_t c, value_t lower)
    {
        return std::log(c - lower);
    }

    value_t inv_transform(value_t uc, value_t lower)
    {
        return std::exp(uc) + lower;
    }
};

TEST_F(lower_fixture, sanity)
{
    value_t x = 1.3;
    value_t l = 0.32;
    EXPECT_DOUBLE_EQ(inv_transform(transform(x, l), l), x);
    EXPECT_DOUBLE_EQ(transform(inv_transform(x, l), l), x);
}

TEST_F(lower_fixture, inv_transform) 
{
    value_t uc = 3.213;
    value_t c = 0;

    constraint.inv_transform(uc, c);
    EXPECT_DOUBLE_EQ(c, inv_transform(uc, expr.get()));
}

TEST_F(lower_fixture, transform) 
{
    value_t c = 5.23;
    value_t uc = 0;
    constraint.transform(c, uc);
    EXPECT_DOUBLE_EQ(uc, transform(c, expr.get()));
}

} // namespace constraint
} // namespace expr
} // namespace ppl
