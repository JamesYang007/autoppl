#include <gtest/gtest.h>
#include <fastad_bits/reverse/core/var.hpp>
#include <fastad_bits/reverse/core/var_view.hpp>
#include <autoppl/util/ad_boost/lower_inv_transform.hpp>

namespace ad {
namespace boost {

struct lower_inv_transform_fixture:
    ::testing::Test
{
protected:
    using value_t = double;
    using scl_var_t = Var<value_t, scl>;
    using vec_var_t = Var<value_t, vec>;
    using scl_var_view_t = VarView<value_t, scl>;
    using vec_var_view_t = VarView<value_t, vec>;
    using ss_transform_t = LowerInvTransformNode<scl_var_view_t, scl_var_view_t>;
    using vs_transform_t = LowerInvTransformNode<vec_var_view_t, scl_var_view_t>;
    using vv_transform_t = LowerInvTransformNode<vec_var_view_t, vec_var_view_t>;

    value_t seed = 2.3142;
    size_t visit = 0;
    size_t refcnt = 2;

    scl_var_t scl_x;
    vec_var_t vec_x;
    scl_var_t scl_lower;
    vec_var_t vec_lower;

    value_t scl_x_c_val;
    Eigen::VectorXd vec_x_c_val;

    ss_transform_t ss_transform;
    vs_transform_t vs_transform;
    vv_transform_t vv_transform;

    Eigen::VectorXd val_buf;
    Eigen::VectorXd adj_buf;

    lower_inv_transform_fixture()
        : scl_x()
        , vec_x(3)
        , scl_lower()
        , vec_lower(3)
        , scl_x_c_val(0)
        , vec_x_c_val(3)
        , ss_transform(scl_x, scl_lower, &scl_x_c_val, &visit, refcnt)
        , vs_transform(vec_x, scl_lower, vec_x_c_val.data(), &visit, refcnt)
        , vv_transform(vec_x, vec_lower, vec_x_c_val.data(), &visit, refcnt)
    {
        scl_x.get() = 1.323;

        auto& vec_x_raw = vec_x.get();
        vec_x_raw[0] = 3.2;
        vec_x_raw[1] = 4.3;
        vec_x_raw[2] = 1.324;

        scl_lower.get() = 0.2314;

        auto& vec_lower_raw = vec_lower.get();
        vec_lower_raw[0] = 3.1;
        vec_lower_raw[1] = 2.3;
        vec_lower_raw[2] = 0.013;

        vec_x_c_val.setZero();

        auto max_pack = ss_transform.bind_cache_size();
        max_pack = max_pack.max(vs_transform.bind_cache_size());
        max_pack = max_pack.max(vv_transform.bind_cache_size());
        val_buf.resize(max_pack(0));
        adj_buf.resize(max_pack(1));
        ss_transform.bind_cache({val_buf.data(), adj_buf.data()});
        vs_transform.bind_cache({val_buf.data(), adj_buf.data()});
        vv_transform.bind_cache({val_buf.data(), adj_buf.data()});
    }
    
};

// Just check that it compiles
TEST_F(lower_inv_transform_fixture, inv_transform) 
{
    lower_inv_transform(scl_x.get(), scl_lower.get(), scl_x_c_val);
    lower_inv_transform(vec_x.get(), scl_lower.get(), vec_x_c_val);
    lower_inv_transform(vec_x.get(), vec_lower.get(), vec_x_c_val);
}

TEST_F(lower_inv_transform_fixture, ss_lower_inv_transform_feval)
{
    value_t actual = 0;
    lower_inv_transform(scl_x.get(), scl_lower.get(), actual);
    value_t res = ss_transform.feval();  
    EXPECT_EQ(visit, 1ul);
    EXPECT_DOUBLE_EQ(res, actual);
    EXPECT_DOUBLE_EQ(res, scl_x_c_val);

    // second time evaluation simulates another AD node in an expression
    // viewing the same resources: check that visit is properly reset.
    res = ss_transform.feval();
    EXPECT_EQ(visit, 0ul);
    EXPECT_DOUBLE_EQ(res, actual);
    EXPECT_DOUBLE_EQ(res, scl_x_c_val);
}

TEST_F(lower_inv_transform_fixture, ss_lower_inv_transform_beval)
{
    value_t actual = std::exp(scl_x.get());

    // evaluate twice to simulate a complicated expression requiring
    // refcnt number of evaluations.
    ss_transform.feval();  
    ss_transform.feval();  
    ss_transform.beval(seed);

    auto& dx = scl_x.get_adj();
    EXPECT_DOUBLE_EQ(dx, seed * actual);
    EXPECT_DOUBLE_EQ(scl_lower.get_adj(), seed);
}

TEST_F(lower_inv_transform_fixture, vs_lower_inv_transform_feval)
{
    Eigen::VectorXd actual(vec_x.size());
    lower_inv_transform(vec_x.get(), scl_lower.get(), actual);
    Eigen::VectorXd res = vs_transform.feval();  
    EXPECT_EQ(visit, 1ul);
    for (int i = 0; i < actual.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), actual(i));
        EXPECT_DOUBLE_EQ(res(i), vec_x_c_val(i));
    }

    // second time evaluation simulates another AD node in an expression
    // viewing the same resources: check that visit is properly reset.
    res = vs_transform.feval();
    EXPECT_EQ(visit, 0ul);
    for (int i = 0; i < actual.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), actual(i));
        EXPECT_DOUBLE_EQ(res(i), vec_x_c_val(i));
    }
}

TEST_F(lower_inv_transform_fixture, vs_lower_inv_transform_beval)
{
    Eigen::VectorXd actual = vec_x.get().array().exp();

    // evaluate twice to simulate a complicated expression requiring
    // refcnt number of evaluations.
    vs_transform.feval();  
    vs_transform.feval();  
    vs_transform.beval(seed);

    auto& dx = vec_x.get_adj();

    for (int i = 0; i < dx.size(); ++i) {
        value_t adj = seed * actual(i);
        EXPECT_DOUBLE_EQ(dx(i), adj);
    }

    EXPECT_DOUBLE_EQ(scl_lower.get_adj(), seed * vec_x.size());
}

TEST_F(lower_inv_transform_fixture, vv_lower_inv_transform_feval)
{
    Eigen::VectorXd actual(vec_x.size());
    lower_inv_transform(vec_x.get(), vec_lower.get(), actual);
    Eigen::VectorXd res = vv_transform.feval();  
    EXPECT_EQ(visit, 1ul);
    for (int i = 0; i < actual.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), actual(i));
        EXPECT_DOUBLE_EQ(res(i), vec_x_c_val(i));
    }

    // second time evaluation simulates another AD node in an expression
    // viewing the same resources: check that visit is properly reset.
    res = vv_transform.feval();
    EXPECT_EQ(visit, 0ul);
    for (int i = 0; i < actual.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), actual(i));
        EXPECT_DOUBLE_EQ(res(i), vec_x_c_val(i));
    }
}

TEST_F(lower_inv_transform_fixture, vv_lower_inv_transform_beval)
{
    Eigen::VectorXd actual = vec_x.get().array().exp();

    // evaluate twice to simulate a complicated expression requiring
    // refcnt number of evaluations.
    vv_transform.feval();  
    vv_transform.feval();  
    vv_transform.beval(seed);

    auto& dx = vec_x.get_adj();

    for (int i = 0; i < dx.size(); ++i) {
        value_t adj = seed * actual(i);
        EXPECT_DOUBLE_EQ(dx(i), adj);

        adj = seed;
        EXPECT_DOUBLE_EQ(vec_lower.get_adj()(i), adj);
    }
}

} // namespace boost
} // namespace ad
