#include "gtest/gtest.h"
#include <fastad>
#include <testutil/base_fixture.hpp>
#include <autoppl/expression/variable/binary.hpp>
#include <autoppl/expression/variable/param.hpp>

namespace ppl {
namespace expr {
namespace var {

//////////////////////////////////////////////////////
// Model with one RV TESTS
//////////////////////////////////////////////////////

struct binary_fixture : 
    base_fixture<double>,
    ::testing::Test
{
protected:
	using scl_binary_t = BinaryNode<ad::math::Add, 
                                      scl_pv_t, 
                                      scl_pv_t>;
	using scl_vec_binary_t = BinaryNode<ad::math::Add, 
                                      scl_pv_t, 
                                      vec_pv_t>;
	using vec_binary_t = BinaryNode<ad::math::Add, 
                                      vec_pv_t, 
                                      vec_pv_t>;
	using mat_binary_t = BinaryNode<ad::math::Add, 
                                      mat_pv_t, 
                                      mat_pv_t>;

    size_t vec_size = 3;
    size_t rows = 2;
    size_t cols = 3;

    scl_p_t scl_x;
    scl_p_t scl_y;
    vec_p_t vec_x;
    vec_p_t vec_y;
    mat_p_t mat_x;
    mat_p_t mat_y;

    scl_binary_t scl_binary;
    scl_vec_binary_t scl_vec_binary;
    vec_binary_t vec_binary;
    mat_binary_t mat_binary;

    binary_fixture()
        : scl_x()
        , scl_y()
        , vec_x(vec_size)
        , vec_y(vec_size)
        , mat_x(rows, cols)
        , mat_y(rows, cols)
        , scl_binary(scl_x, scl_y)
        , scl_vec_binary(scl_x, vec_y)
        , vec_binary(vec_x, vec_y)
        , mat_binary(mat_x, mat_y)
    {
        offset_pack_t offset;
        scl_x.activate(offset);
        vec_x.activate(offset);
        mat_x.activate(offset);

        offset.uc_offset = mat_x.size();
        scl_y.activate(offset);
        vec_y.activate(offset);
        mat_y.activate(offset);

        val_buf.resize(100);
        val_buf[0] = 2;
        val_buf[1] = 3;
        val_buf[2] = 0;
        val_buf[3] = -2;
        val_buf[4] = -1;
        val_buf[5] = 4;
        val_buf[6] = 3;
        val_buf[7] = 4.2;
        val_buf[8] = -2;
        val_buf[9] = -10;

        bind(scl_binary);
        bind(scl_vec_binary);
        bind(vec_binary);
        bind(mat_binary);

        ptr_pack.uc_val = val_buf.data();
    }
};

//////////////////////////////////////////////////////
// binary Node TESTS
//////////////////////////////////////////////////////

TEST_F(binary_fixture, scl_get)
{
	EXPECT_DOUBLE_EQ(scl_binary.eval(), 
                     val_buf[scl_x.offset().uc_offset] +
                     val_buf[scl_y.offset().uc_offset]);
}

TEST_F(binary_fixture, scl_size)
{
	EXPECT_EQ(scl_binary.size(), 1ul);
}

TEST_F(binary_fixture, scl_ad)
{
    auto expr = ad::bind(scl_binary.ad(ptr_pack));
	EXPECT_DOUBLE_EQ(ad::evaluate(expr), 
                     val_buf[scl_x.offset().uc_offset] + 
                     val_buf[scl_y.offset().uc_offset]);
}

TEST_F(binary_fixture, scl_vec_get)
{
    Eigen::VectorXd res = scl_vec_binary.eval();
    for (size_t i = 0; i < scl_vec_binary.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), 
                         val_buf[scl_x.offset().uc_offset] + 
                         val_buf[vec_y.offset().uc_offset + i]);
    }
}

TEST_F(binary_fixture, scl_vec_size)
{
	EXPECT_EQ(scl_vec_binary.size(), vec_size);
}

TEST_F(binary_fixture, scl_vec_ad)
{
    auto expr = ad::bind(scl_vec_binary.ad(ptr_pack));
    Eigen::VectorXd res = ad::evaluate(expr);
    for (size_t i = 0; i < scl_vec_binary.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), 
                         val_buf[scl_x.offset().uc_offset] + 
                         val_buf[vec_y.offset().uc_offset + i]);
    }
}

TEST_F(binary_fixture, vec_get)
{
    Eigen::VectorXd res = vec_binary.eval();
    for (size_t i = 0; i < vec_binary.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), 
                         val_buf[vec_x.offset().uc_offset + i] + 
                         val_buf[vec_y.offset().uc_offset + i]);
    }
}

TEST_F(binary_fixture, vec_size)
{
	EXPECT_EQ(vec_binary.size(), vec_size);
}

TEST_F(binary_fixture, vec_ad)
{
    auto expr = ad::bind(vec_binary.ad(ptr_pack));
    Eigen::VectorXd res = ad::evaluate(expr);
    for (size_t i = 0; i < vec_binary.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), 
                         val_buf[vec_x.offset().uc_offset + i] + 
                         val_buf[vec_y.offset().uc_offset + i]);
    }
}

TEST_F(binary_fixture, mat_get)
{
    Eigen::MatrixXd res = mat_binary.eval();
    for (size_t i = 0; i < mat_binary.rows(); ++i) {
        for (size_t j = 0; j < mat_binary.cols(); ++j) {
            EXPECT_DOUBLE_EQ(res(i,j), 
                             val_buf[mat_x.offset().uc_offset + i + j*mat_binary.rows()] + 
                             val_buf[mat_y.offset().uc_offset + i + j*mat_binary.rows()]);
        }
    }
}

TEST_F(binary_fixture, mat_size)
{
	EXPECT_EQ(mat_binary.size(), rows * cols);
}

TEST_F(binary_fixture, mat_ad)
{
    auto expr = ad::bind(mat_binary.ad(ptr_pack));
    Eigen::MatrixXd res = ad::evaluate(expr);
    for (size_t i = 0; i < mat_binary.rows(); ++i) {
        for (size_t j = 0; j < mat_binary.cols(); ++j) {
            EXPECT_DOUBLE_EQ(res(i,j), 
                             val_buf[mat_x.offset().uc_offset + i + j*mat_binary.rows()] + 
                             val_buf[mat_y.offset().uc_offset + i + j*mat_binary.rows()]);
        }
    }
}

} // namespace var
} // namespace expr
} // namespace ppl
