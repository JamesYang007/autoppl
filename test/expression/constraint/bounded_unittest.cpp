#include <gtest/gtest.h>
#include <testutil/base_fixture.hpp>
#include <autoppl/expression/constraint/bounded.hpp>

namespace ppl {
namespace expr {
namespace constraint {

struct bounded_fixture:
    base_fixture<double>,
    ::testing::Test
{
protected:
    using bounded_t = Bounded<scl_dv_t, scl_dv_t>;
    scl_d_t lower;
    scl_d_t upper;
    bounded_t constraint;

    bounded_fixture()
        : lower(0.32)
        , upper(3.15523)
        , constraint(lower, upper)
    {}

    value_t logit(value_t u) {
        return std::log(u / (1. - u));
    }

    value_t inv_logit(value_t v) {
        return 1. / (1. + std::exp(-v));
    }

    value_t transform(value_t c, value_t a, value_t b)
    {
        return logit((c-a)/(b-a));
    }

    value_t inv_transform(value_t uc, value_t a, value_t b)
    {
        return a + (b-a) * inv_logit(uc);
    }
};

TEST_F(bounded_fixture, sanity)
{
    value_t x = 1.3;
    value_t a = lower.get();
    value_t b = upper.get();
    EXPECT_DOUBLE_EQ(inv_transform(transform(x, a, b), a, b), x);
    EXPECT_DOUBLE_EQ(transform(inv_transform(x, a, b), a, b), x);
}

TEST_F(bounded_fixture, inv_transform) 
{
    value_t uc = 3.213;
    value_t c = 0;
    value_t a = lower.get();
    value_t b = upper.get();

    constraint.inv_transform(uc, c);
    EXPECT_DOUBLE_EQ(c, inv_transform(uc, a, b));
}

TEST_F(bounded_fixture, transform) 
{
    value_t c = 2.23;
    value_t uc = 0;
    value_t a = lower.get();
    value_t b = upper.get();
    constraint.transform(c, uc);
    EXPECT_DOUBLE_EQ(uc, transform(c, a, b));
}

} // namespace constraint
} // namespace expr
} // namespace ppl
