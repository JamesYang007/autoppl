#include <gtest/gtest.h>
#include <fastad_bits/reverse/core/var.hpp>
#include <fastad_bits/reverse/core/var_view.hpp>
#include <autoppl/expression/constraint/pos_def.hpp>
#include <autoppl/util/ad_boost/cov_inv_transform.hpp>

namespace ad {
namespace boost {

struct cov_inv_transform_fixture:
    ::testing::Test
{
protected:
    using value_t = double;
    using pos_def_t = ppl::expr::constraint::PosDef;
    using vec_var_t = Var<value_t, vec>;
    using vec_var_view_t = VarView<value_t, vec>;
    using transform_t = CovInvTransformNode<vec_var_view_t>;
    using logj_transform_t = LogJCovInvTransformNode<vec_var_view_t>;

    value_t seed = 2.3142;
    size_t rows = 3;
    size_t visit = 0;
    size_t refcnt = 2;

    pos_def_t pos_def;
    vec_var_t x;
    Eigen::MatrixXd lower;
    Eigen::MatrixXd val;

    transform_t transform_expr;
    logj_transform_t logj_transform_expr;

    Eigen::VectorXd val_buf;
    Eigen::VectorXd adj_buf;

    cov_inv_transform_fixture()
        : x(pos_def_t::size(rows))
        , lower(rows, rows)
        , val(rows, rows)
        , transform_expr(x, lower.data(), val.data(), rows, &visit, refcnt)
        , logj_transform_expr(x, rows)
    {
        auto& x_raw = x.get();
        x_raw[0] = 1.;
        x_raw[1] = 0.2;
        x_raw[2] = -1.;
        x_raw[3] = 3.;
        x_raw[4] = 2.;
        x_raw[5] = 11.;

        lower.setZero();
        val.setZero();

        auto max_pack = transform_expr.bind_cache_size();
        max_pack = max_pack.max(logj_transform_expr.bind_cache_size());
        val_buf.resize(max_pack(0));
        adj_buf.resize(max_pack(1));
        transform_expr.bind_cache({val_buf.data(), adj_buf.data()});
        logj_transform_expr.bind_cache({val_buf.data(), adj_buf.data()});
    }
};

TEST_F(cov_inv_transform_fixture, cov_inv_transform_feval)
{
    Eigen::MatrixXd lower(rows, rows);
    Eigen::MatrixXd actual(rows, rows);
    lower.setZero();
    pos_def.inv_transform(lower, x.get(), actual);

    Eigen::MatrixXd res;

    res = transform_expr.feval();  
    EXPECT_EQ(visit, 1ul);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < rows; ++j) {
            EXPECT_DOUBLE_EQ(res(i,j), actual(i,j));
        }
    }

    // second time evaluation simulates another AD node in an expression
    // viewing the same resources: check that visit is properly reset.
    res = transform_expr.feval();
    EXPECT_EQ(visit, 0ul);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < rows; ++j) {
            EXPECT_DOUBLE_EQ(res(i,j), actual(i,j));
        }
    }
}

TEST_F(cov_inv_transform_fixture, cov_inv_transform_beval)
{
    // evaluate twice to simulate a complicated expression requiring
    // refcnt number of evaluations.
    transform_expr.feval();  
    transform_expr.feval();  
    transform_expr.beval(seed);

    auto& z = lower;
    auto& dx = x.get_adj();
    
    Eigen::VectorXd actual(dx.size());
    actual.setZero();

    auto adj = [&](const auto& z, size_t p, size_t q, size_t k) {
        if ((rows * q + p) - (q * (q+1))/2 == k) {
            if (p == q) return z(p, p);
            else if (p > q) return 1.;
            else return 0.;
        } else {
            return 0.;
        }
    };

    auto update = [&](size_t i, size_t j) {
        for (int k = 0; k < actual.size(); ++k) {
            for (size_t l = 0; l < rows; ++l) {
                actual(k) += seed * (adj(z,i,l,k) * z(j,l) + z(i,l) * adj(z,j,l,k));
            }
        }
    };

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < rows; ++j) {
            update(i, j);
        }
    }

    EXPECT_EQ(dx.size(), actual.size());
    for (int k = 0; k < actual.size(); ++k) {
        EXPECT_DOUBLE_EQ(dx(k), actual(k));
    }
}

TEST_F(cov_inv_transform_fixture, logj_cov_inv_transform_feval)
{
    transform_expr.feval();
    transform_expr.feval();
    value_t res = logj_transform_expr.feval();

    value_t actual = 0;
    for (size_t k = 0; k < rows; ++k) {
        actual += (rows - k + 1) * std::log(lower(k,k));
    }

    EXPECT_DOUBLE_EQ(res, actual);
}

TEST_F(cov_inv_transform_fixture, logj_cov_inv_transform_beval)
{
    transform_expr.feval();
    transform_expr.feval();
    logj_transform_expr.feval();
    logj_transform_expr.beval(seed);

    size_t k = 0;
    for (size_t j = 0; j < rows; ++j) {
        EXPECT_DOUBLE_EQ(x.get_adj()(k), seed * (rows - j + 1));
        ++k;
        for (size_t i = j+1; i < rows; ++i, ++k) {
            EXPECT_DOUBLE_EQ(x.get_adj()(k), 0.);
        }
    }
    EXPECT_EQ(k, x.size());
}

} // namespace boost
} // namespace ad
