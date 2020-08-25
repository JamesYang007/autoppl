#include <gtest/gtest.h>
#include <testutil/base_fixture.hpp>
#include <autoppl/expression/constraint/pos_def.hpp>

namespace ppl {
namespace expr {
namespace constraint {

struct pos_def_fixture:
    base_fixture<double>,
    ::testing::Test
{
protected:
    value_t tol = 1e-15;
};

TEST_F(pos_def_fixture, size) 
{
    EXPECT_EQ(PosDef::size(1), 1ul);
    EXPECT_EQ(PosDef::size(2), 3ul);
    EXPECT_EQ(PosDef::size(3), 6ul);
    EXPECT_EQ(PosDef::size(4), 10ul);
    EXPECT_EQ(PosDef::size(5), 15ul);
}

TEST_F(pos_def_fixture, inv_transform) 
{
    Eigen::MatrixXd lower(3,3);
    Eigen::VectorXd uc(6);
    Eigen::MatrixXd c(3,3);

    lower.setZero();
    uc[0] = 1.;
    uc[1] = 2.;
    uc[2] = 3.;
    uc[3] = 2.;
    uc[4] = 5.;
    uc[5] = -1.;

    PosDef::inv_transform(lower, uc, c);

    Eigen::MatrixXd z(3,3);
    z.setZero();
    z(0,0) = std::exp(uc[0]);
    z(1,0) = uc[1];
    z(2,0) = uc[2];
    z(1,1) = std::exp(uc[3]);
    z(2,1) = uc[4];
    z(2,2) = std::exp(uc[5]);
    Eigen::MatrixXd actual = z * z.transpose();

    for (int i = 0; i < c.rows(); ++i) {
        for (int j = 0; j < c.cols(); ++j) {
            EXPECT_DOUBLE_EQ(c(i,j), actual(i,j));
        }
    }
}

TEST_F(pos_def_fixture, transform) 
{
    Eigen::MatrixXd lower(3,3);
    Eigen::VectorXd uc(6);
    Eigen::MatrixXd c(3,3);

    lower.setZero();
    uc[0] = 1.;
    uc[1] = 2.;
    uc[2] = 3.;
    uc[3] = 2.;
    uc[4] = 5.;
    uc[5] = -1.;

    PosDef::inv_transform(lower, uc, c);
    
    Eigen::VectorXd uc_expected(6);
    PosDef::transform(c, uc_expected);

    for (int i = 0; i < uc.size(); ++i) {
        EXPECT_NEAR(uc(i), uc_expected(i), 7*tol);
    }
}

} // namespace constraint
} // namespace expr
} // namespace ppl
