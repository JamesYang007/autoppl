#include <gtest/gtest.h>
#include <testutil/base_fixture.hpp>
#include <fastad>
#include <autoppl/expression/op_overloads.hpp>
#include <autoppl/expression/variable/glue.hpp>
#include <autoppl/expression/variable/op_eq.hpp>
#include <autoppl/expression/variable/binary.hpp>

namespace ppl {
namespace expr {
namespace var {

struct glue_fixture:
    base_fixture<double>,
    ::testing::Test
{
protected:
    using scl_binary_t = BinaryNode<ad::math::Add,
                                    scl_pv_t,
                                    scl_c_t>;
	using scl_op_eq_t = OpEqNode<Eq, 
                                 scl_tpv_t, 
                                 scl_binary_t>;
	using vec_op_eq_t = OpEqNode<AddEq, 
                                 vec_tpv_t, 
                                 vec_tpv_t>;

    scl_tp_t scl_tp;
    scl_p_t scl_p;
    vec_tp_t vec_tp;

    scl_op_eq_t scl_op_eq;
    vec_op_eq_t vec_op_eq;

    Eigen::VectorXd uc_val;
    Eigen::VectorXd tp_val;

    glue_fixture()
        : scl_tp()
        , scl_p()
        , vec_tp(3)
        , scl_op_eq(scl_tp, scl_p + 1.)
        , vec_op_eq(vec_tp, vec_tp)
    {
        uc_val.resize(scl_p.size());
        tp_val.resize(scl_tp.size() + vec_tp.size());

        uc_val[0] = 0.3;
        tp_val[1] = 0.2;
        tp_val[2] = -3.2;
        tp_val[3] = 4.1;

        offset_pack_t offset;
        scl_p.activate(offset);
        scl_tp.activate(offset);
        vec_tp.activate(offset);

        ptr_pack.uc_val = uc_val.data();
        ptr_pack.tp_val = tp_val.data();

        scl_op_eq.bind(ptr_pack);
        vec_op_eq.bind(ptr_pack);
    }
};

TEST_F(glue_fixture, glue_eval)
{
    Eigen::VectorXd vec_orig = tp_val.tail(vec_tp.size());

    auto glue = (scl_op_eq, vec_op_eq);

    Eigen::VectorXd res = glue.eval();
    Eigen::VectorXd vec_new = tp_val.tail(vec_tp.size());

    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), vec_new(i));
    }

    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec_new(i), 2*vec_orig(i));
    }

    EXPECT_DOUBLE_EQ(tp_val[0], uc_val[0] + 1);
}

TEST_F(glue_fixture, glue_ad)
{
    Eigen::VectorXd vec_orig = tp_val.tail(vec_tp.size());

    auto glue = (scl_op_eq, vec_op_eq);
    auto expr = ad::bind(glue.ad(ptr_pack));

    Eigen::VectorXd res = ad::evaluate(expr);
    Eigen::VectorXd vec_new = tp_val.tail(vec_tp.size());

    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), vec_new(i));
    }

    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec_new(i), 2*vec_orig(i));
    }

    EXPECT_DOUBLE_EQ(tp_val[0], uc_val[0] + 1);
}

} // namespace var
} // namespace expr
} // namespace ppl
