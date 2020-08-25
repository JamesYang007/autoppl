#include <gtest/gtest.h>
#include <testutil/base_fixture.hpp>
#include <fastad>
#include <autoppl/expression/op_overloads.hpp>
#include <autoppl/expression/variable/for_each.hpp>
#include <autoppl/expression/variable/op_eq.hpp>
#include <autoppl/expression/variable/binary.hpp>
#include <autoppl/util/iterator/counting_iterator.hpp>

namespace ppl {
namespace expr {
namespace var {

struct for_each_fixture:
    base_fixture<double>,
    ::testing::Test
{
protected:
    scl_tp_t scl_tp;
    scl_p_t scl_p;

    Eigen::VectorXd uc_val;
    Eigen::VectorXd tp_val;

    for_each_fixture()
        : scl_tp()
        , scl_p()
    {
        uc_val.resize(scl_p.size());
        tp_val.resize(scl_tp.size());

        uc_val[0] = 0.3;
        tp_val[0] = 0.21;

        offset_pack_t offset;
        scl_p.activate(offset);
        scl_tp.activate(offset);

        ptr_pack.uc_val = uc_val.data();
        ptr_pack.tp_val = tp_val.data();
    }
};

TEST_F(for_each_fixture, for_each_eval)
{
    value_t uc_orig = uc_val[0];
    value_t tp_orig = tp_val[0];
    auto expr = for_each(util::counting_iterator<>(0),
                         util::counting_iterator<>(3),
                         [&](size_t) { return scl_tp += scl_p * 2.; });
    expr.bind(ptr_pack);
    value_t res = expr.eval();
    EXPECT_DOUBLE_EQ(res, 6. * uc_orig + tp_orig);
}

TEST_F(for_each_fixture, for_each_ad)
{
    value_t uc_orig = uc_val[0];
    value_t tp_orig = tp_val[0];
    auto expr = for_each(util::counting_iterator<>(0),
                         util::counting_iterator<>(3),
                         [&](size_t) { return scl_tp += scl_p * 2.; });
    expr.bind(ptr_pack);
    auto ad_expr = ad::bind(expr.ad(ptr_pack));
    value_t res = ad::evaluate(ad_expr);
    EXPECT_DOUBLE_EQ(res, 6. * uc_orig + tp_orig);
}

} // namespace var
} // namespace expr
} // namespace ppl
