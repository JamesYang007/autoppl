#include <gtest/gtest.h>
#include <fastad_bits/reverse/core/var.hpp>
#include <fastad_bits/reverse/core/var_view.hpp>
#include <autoppl/util/ad_boost/bounded_inv_transform.hpp>

namespace ad {
namespace boost {

struct bounded_inv_transform_fixture:
    ::testing::Test
{
protected:
    using value_t = double;
    using scl_var_t = Var<value_t, scl>;
    using vec_var_t = Var<value_t, vec>;
    using scl_var_view_t = VarView<value_t, scl>;
    using vec_var_view_t = VarView<value_t, vec>;
    using sss_transform_t = BoundedInvTransformNode<scl_var_view_t, 
                                                    scl_var_view_t, 
                                                    scl_var_view_t>;
    using vss_transform_t = BoundedInvTransformNode<vec_var_view_t, 
                                                    scl_var_view_t,
                                                    scl_var_view_t>;
    using vsv_transform_t = BoundedInvTransformNode<vec_var_view_t, 
                                                    scl_var_view_t,
                                                    vec_var_view_t>;
    using vvs_transform_t = BoundedInvTransformNode<vec_var_view_t, 
                                                    vec_var_view_t,
                                                    scl_var_view_t>;
    using vvv_transform_t = BoundedInvTransformNode<vec_var_view_t, 
                                                    vec_var_view_t,
                                                    vec_var_view_t>;

    using sss_logj_t = LogJBoundedInvTransformNode<scl_var_view_t, 
                                                   scl_var_view_t, 
                                                   scl_var_view_t>;
    using vss_logj_t = LogJBoundedInvTransformNode<vec_var_view_t, 
                                                   scl_var_view_t,
                                                   scl_var_view_t>;
    using vsv_logj_t = LogJBoundedInvTransformNode<vec_var_view_t, 
                                                   scl_var_view_t,
                                                   vec_var_view_t>;
    using vvs_logj_t = LogJBoundedInvTransformNode<vec_var_view_t, 
                                                   vec_var_view_t,
                                                   scl_var_view_t>;
    using vvv_logj_t = LogJBoundedInvTransformNode<vec_var_view_t, 
                                                   vec_var_view_t,
                                                   vec_var_view_t>;

    value_t tol = 2e-15;

    value_t seed = 2.3142;
    size_t visit = 0;
    size_t refcnt = 2;

    scl_var_t scl_x;
    vec_var_t vec_x;
    scl_var_t scl_lower;
    vec_var_t vec_lower;
    scl_var_t scl_upper;
    vec_var_t vec_upper;

    value_t scl_x_c_val;
    Eigen::VectorXd vec_x_c_val;

    sss_transform_t sss_transform;
    vss_transform_t vss_transform;
    vsv_transform_t vsv_transform;
    vvs_transform_t vvs_transform;
    vvv_transform_t vvv_transform;

    sss_logj_t sss_logj;
    vss_logj_t vss_logj;
    vsv_logj_t vsv_logj;
    vvs_logj_t vvs_logj;
    vvv_logj_t vvv_logj;

    Eigen::VectorXd val_buf;
    Eigen::VectorXd adj_buf;

    bounded_inv_transform_fixture()
        : scl_x()
        , vec_x(3)
        , scl_lower(0.2314)
        , vec_lower(3)
        , scl_upper(3.5561)
        , vec_upper(3)
        , scl_x_c_val(0)
        , vec_x_c_val(3)
        , sss_transform(scl_x, scl_lower, scl_upper, &scl_x_c_val, &visit, refcnt)
        , vss_transform(vec_x, scl_lower, scl_upper, vec_x_c_val.data(), &visit, refcnt)
        , vsv_transform(vec_x, scl_lower, vec_upper, vec_x_c_val.data(), &visit, refcnt)
        , vvs_transform(vec_x, vec_lower, scl_upper, vec_x_c_val.data(), &visit, refcnt)
        , vvv_transform(vec_x, vec_lower, vec_upper, vec_x_c_val.data(), &visit, refcnt)
        , sss_logj(scl_x, scl_lower, scl_upper, &scl_x_c_val)
        , vss_logj(vec_x, scl_lower, scl_upper, vec_x_c_val.data())
        , vsv_logj(vec_x, scl_lower, vec_upper, vec_x_c_val.data())
        , vvs_logj(vec_x, vec_lower, scl_upper, vec_x_c_val.data())
        , vvv_logj(vec_x, vec_lower, vec_upper, vec_x_c_val.data())
    {
        scl_x.get() = -0.31;
        vec_x.get() << 0.32, 2.1, 1.4;
        vec_lower.get() << 0.3, 0.03, -1.32;
        vec_upper.get() << 1.3, 3.5, 2.1;

        vec_x_c_val.setZero();

        auto max_pack = sss_transform.bind_cache_size();
        max_pack = max_pack.max(vss_transform.bind_cache_size());
        max_pack = max_pack.max(vsv_transform.bind_cache_size());
        max_pack = max_pack.max(vvs_transform.bind_cache_size());
        max_pack = max_pack.max(vvv_transform.bind_cache_size());
        max_pack = max_pack.max(sss_logj.bind_cache_size());
        max_pack = max_pack.max(vss_logj.bind_cache_size());
        max_pack = max_pack.max(vsv_logj.bind_cache_size());
        max_pack = max_pack.max(vvs_logj.bind_cache_size());
        max_pack = max_pack.max(vvv_logj.bind_cache_size());

        val_buf.resize(max_pack(0));
        adj_buf.resize(max_pack(1));
        
        sss_transform.bind_cache({val_buf.data(), adj_buf.data()});
        vss_transform.bind_cache({val_buf.data(), adj_buf.data()});
        vsv_transform.bind_cache({val_buf.data(), adj_buf.data()});
        vvs_transform.bind_cache({val_buf.data(), adj_buf.data()});
        vvv_transform.bind_cache({val_buf.data(), adj_buf.data()});
        sss_logj.bind_cache({val_buf.data(), adj_buf.data()});
        vss_logj.bind_cache({val_buf.data(), adj_buf.data()});
        vsv_logj.bind_cache({val_buf.data(), adj_buf.data()});
        vvs_logj.bind_cache({val_buf.data(), adj_buf.data()});
        vvv_logj.bind_cache({val_buf.data(), adj_buf.data()});
    }

    template <class T>
    auto inv_logit(const T& y)
    {
        using std::exp;
        using Eigen::exp;
        return 1. / (1. + exp(-util::to_array(y)));
    }

    template <class T>
    auto dinv_logit(const T& y) {
        auto il = inv_logit(y);
        return il * (1. - il);
    }

    template <class T>
    auto ddinv_logit(const T& y) {
        auto il = inv_logit(y);
        auto dil = dinv_logit(y);
        return dil * (1. - il) - il * dil;
    }

    template <class T, class Lower, class Upper>
    auto dinv_transform(const T& y,
                        const Lower& lower,
                        const Upper& upper) {
        auto a = util::to_array(lower);
        auto b = util::to_array(upper);
        return (b-a) * dinv_logit(y);
    }

    template <class T>
    auto dinv_transform_dl(const T& y) {
        return 1 - inv_logit(y);
    }

    template <class T>
    auto dinv_transform_du(const T& y) {
        return inv_logit(y);
    }

    template <class T, class Lower, class Upper>
    auto ddinv_transform(const T& y,
                         const Lower& lower,
                         const Upper& upper) {
        auto a = util::to_array(lower);
        auto b = util::to_array(upper);
        return (b-a) * ddinv_logit(y);
    }

    template <class T, class Lower, class Upper>
    auto logj_inv_transform(const T& y,
                            const Lower& lower,
                            const Upper& upper) {
        using std::log;
        using Eigen::log;
        auto inner = log(dinv_transform(y, lower, upper));
        if constexpr (std::is_arithmetic_v<std::decay_t<decltype(inner)>>) {
            return inner;
        } else {
            return inner.sum();
        }
    }

    template <class T, class Lower, class Upper>
    auto dlogj_inv_transform(const T& y,    
                             const Lower& lower,
                             const Upper& upper) {
        return ddinv_transform(y, lower, upper) / 
            dinv_transform(y, lower, upper);
    }

    template <class Lower, class Upper>
    auto dlogj_inv_transform_dl(const Lower& lower,
                                const Upper& upper) {
        auto a = util::to_array(lower);
        auto b = util::to_array(upper);
        return -1. / (b - a);
    }

    template <class Lower, class Upper>
    auto dlogj_inv_transform_du(const Lower& lower,
                                const Upper& upper) {
        return -dlogj_inv_transform_dl(lower, upper);
    }

    template <class LowerType
            , class UpperType
            , class TransformType>
    void test_vec_x_feval(const LowerType& lower,
                          const UpperType& upper,
                          TransformType& transform) 
    {
        Eigen::VectorXd actual(vec_x.size());
        bounded_inv_transform(vec_x.get(), lower.get(), upper.get(), actual);
        Eigen::VectorXd res = transform.feval();  
        EXPECT_EQ(visit, 1ul);
        for (int i = 0; i < actual.size(); ++i) {
            EXPECT_DOUBLE_EQ(res(i), actual(i));
            EXPECT_DOUBLE_EQ(res(i), vec_x_c_val(i));
        }

        // second time evaluation simulates another AD node in an expression
        // viewing the same resources: check that visit is properly reset.
        res = transform.feval();  
        EXPECT_EQ(visit, 0ul);
        for (int i = 0; i < actual.size(); ++i) {
            EXPECT_DOUBLE_EQ(res(i), actual(i));
            EXPECT_DOUBLE_EQ(res(i), vec_x_c_val(i));
        }
    }

    template <class LowerType
            , class UpperType
            , class TransformType>
    void test_vec_x_beval(const LowerType& lower,
                          const UpperType& upper,
                          TransformType& transform) 
    {
        Eigen::VectorXd actual = dinv_transform(vec_x.get(), lower.get(), upper.get());

        transform.feval();  
        transform.feval();  
        transform.beval(seed);

        auto& dx = vec_x.get_adj();
        for (int i = 0; i < dx.size(); ++i) {
            value_t adj = seed * actual(i);
            EXPECT_NEAR(dx(i), adj, tol);
        }

        Eigen::VectorXd dl = dinv_transform_dl(vec_x.get());
        if constexpr (util::is_scl_v<LowerType>) {
            EXPECT_NEAR(lower.get_adj(), seed * dl.sum(), tol);
        } else {
            for (int i = 0; i < dl.size(); ++i) {
                value_t adj = seed * dl(i);
                EXPECT_NEAR(lower.get_adj()(i), adj, tol);
            }
        }

        Eigen::VectorXd du = dinv_transform_du(vec_x.get());
        if constexpr (util::is_scl_v<UpperType>) {
            EXPECT_NEAR(upper.get_adj(), seed * du.sum(), tol);
        } else {
            for (int i = 0; i < du.size(); ++i) {
                value_t adj = seed * du(i);
                EXPECT_NEAR(upper.get_adj()(i), adj, tol);
            }
        }
    }

    template <class LowerType
            , class UpperType
            , class TransformType
            , class LogJType>
    void test_vec_x_logj_feval(const LowerType& lower,
                               const UpperType& upper,
                               TransformType& transform,
                               LogJType& logj)  
    {
        value_t actual = logj_inv_transform(vec_x.get(), lower.get(), upper.get());

        transform.feval();
        transform.feval();
        value_t res = logj.feval();

        EXPECT_NEAR(res, actual, tol);
    }

    template <class LowerType
            , class UpperType
            , class TransformType
            , class LogJType>
    void test_vec_x_logj_beval(const LowerType& lower,
                               const UpperType& upper,
                               TransformType& transform,
                               LogJType& logj)  
    {
        Eigen::VectorXd actual = dlogj_inv_transform(vec_x.get(), lower.get(), upper.get());

        transform.feval();
        transform.feval();
        logj.feval();
        logj.beval(seed);
        
        auto& dx = vec_x.get_adj();

        for (int i = 0; i < dx.size(); ++i) {
            EXPECT_NEAR(dx(i), seed * actual(i), tol);
        }

        auto& dl = lower.get_adj();
        auto& du = upper.get_adj();
        auto adj = seed * dlogj_inv_transform_dl(lower.get(), upper.get());

        if constexpr (util::is_scl_v<LowerType> &&
                      util::is_scl_v<UpperType>) {
            EXPECT_DOUBLE_EQ(dl, vec_x.size() * adj);
            EXPECT_DOUBLE_EQ(du, vec_x.size() * -adj);

        } else if constexpr (util::is_scl_v<LowerType>) {
            EXPECT_DOUBLE_EQ(dl, adj.sum());
            for (int i = 0; i < du.size(); ++i) {
                EXPECT_DOUBLE_EQ(du(i), -adj(i));
            }

        } else if constexpr (util::is_scl_v<UpperType>) {
            EXPECT_DOUBLE_EQ(du, -adj.sum());
            for (int i = 0; i < dl.size(); ++i) {
                EXPECT_DOUBLE_EQ(dl(i), adj(i));
            }

        } else {
            for (int i = 0; i < du.size(); ++i) {
                EXPECT_DOUBLE_EQ(dl(i), adj(i));
                EXPECT_DOUBLE_EQ(du(i), -adj(i));
            }
        }

    }
};

TEST_F(bounded_inv_transform_fixture, sss_bounded_inv_transform_feval)
{
    value_t actual = 0;
    bounded_inv_transform(scl_x.get(), scl_lower.get(), scl_upper.get(), actual);
    value_t res = sss_transform.feval();  
    EXPECT_EQ(visit, 1ul);
    EXPECT_DOUBLE_EQ(res, actual);
    EXPECT_DOUBLE_EQ(res, scl_x_c_val);

    // second time evaluation simulates another AD node in an expression
    // viewing the same resources: check that visit is properly reset.
    res = sss_transform.feval();  
    EXPECT_EQ(visit, 0ul);
    EXPECT_DOUBLE_EQ(res, actual);
    EXPECT_DOUBLE_EQ(res, scl_x_c_val);
}

TEST_F(bounded_inv_transform_fixture, sss_bounded_inv_transform_beval)
{
    value_t actual = dinv_transform(scl_x.get(), scl_lower.get(), scl_upper.get());

    // evaluate twice to simulate a complicated expression requiring
    // refcnt number of evaluations.
    sss_transform.feval();  
    sss_transform.feval();  
    sss_transform.beval(seed);

    auto& dx = scl_x.get_adj();
    EXPECT_DOUBLE_EQ(dx, seed * actual);
    EXPECT_DOUBLE_EQ(scl_lower.get_adj(), 
            seed * dinv_transform_dl(scl_x.get()));
    EXPECT_DOUBLE_EQ(scl_upper.get_adj(), 
            seed * dinv_transform_du(scl_x.get()));
}

TEST_F(bounded_inv_transform_fixture, vss_bounded_inv_transform_feval)
{
    test_vec_x_feval(scl_lower, scl_upper, vss_transform);
}

TEST_F(bounded_inv_transform_fixture, vss_bounded_inv_transform_beval)
{
    test_vec_x_beval(scl_lower, scl_upper, vss_transform);
}

TEST_F(bounded_inv_transform_fixture, vsv_bounded_inv_transform_feval)
{
    test_vec_x_feval(scl_lower, vec_upper, vsv_transform);
}

TEST_F(bounded_inv_transform_fixture, vsv_bounded_inv_transform_beval)
{
    test_vec_x_beval(scl_lower, vec_upper, vsv_transform);
}

TEST_F(bounded_inv_transform_fixture, vvs_bounded_inv_transform_feval)
{
    test_vec_x_feval(vec_lower, scl_upper, vvs_transform);
}

TEST_F(bounded_inv_transform_fixture, vvs_bounded_inv_transform_beval)
{
    test_vec_x_beval(vec_lower, scl_upper, vvs_transform);
}

TEST_F(bounded_inv_transform_fixture, vvv_bounded_inv_transform_feval)
{
    test_vec_x_feval(vec_lower, vec_upper, vvv_transform);
}

TEST_F(bounded_inv_transform_fixture, vvv_bounded_inv_transform_beval)
{
    test_vec_x_beval(vec_lower, vec_upper, vvv_transform);
}

// LogJ TESTS

TEST_F(bounded_inv_transform_fixture, logj_sss_bounded_inv_transform_feval)
{
    value_t actual = logj_inv_transform(scl_x.get(), scl_lower.get(), scl_upper.get());

    sss_transform.feval();
    sss_transform.feval();
    value_t res = sss_logj.feval();

    EXPECT_NEAR(res, actual, tol);
}

TEST_F(bounded_inv_transform_fixture, logj_sss_bounded_inv_transform_beval)
{
    value_t actual = dlogj_inv_transform(scl_x.get(), scl_lower.get(), scl_upper.get());

    sss_transform.feval();
    sss_transform.feval();
    sss_logj.feval();
    sss_logj.beval(seed);
    
    value_t dx = scl_x.get_adj();
    EXPECT_DOUBLE_EQ(dx, seed * actual);
    EXPECT_DOUBLE_EQ(scl_lower.get_adj(), 
            seed * dlogj_inv_transform_dl(scl_lower.get(), scl_upper.get()));
    EXPECT_DOUBLE_EQ(scl_upper.get_adj(), 
            seed * dlogj_inv_transform_du(scl_lower.get(), scl_upper.get()));
}

TEST_F(bounded_inv_transform_fixture, logj_vss_bounded_inv_transform_feval)
{
    test_vec_x_logj_feval(scl_lower, scl_upper, vss_transform, vss_logj);
}

TEST_F(bounded_inv_transform_fixture, logj_vss_bounded_inv_transform_beval)
{
    test_vec_x_logj_beval(scl_lower, scl_upper, vss_transform, vss_logj);
}

TEST_F(bounded_inv_transform_fixture, logj_vsv_bounded_inv_transform_feval)
{
    test_vec_x_logj_feval(scl_lower, vec_upper, vsv_transform, vsv_logj);
}

TEST_F(bounded_inv_transform_fixture, logj_vsv_bounded_inv_transform_beval)
{
    test_vec_x_logj_beval(scl_lower, vec_upper, vsv_transform, vsv_logj);
}

TEST_F(bounded_inv_transform_fixture, logj_vvs_bounded_inv_transform_feval)
{
    test_vec_x_logj_feval(vec_lower, scl_upper, vvs_transform, vvs_logj);
}

TEST_F(bounded_inv_transform_fixture, logj_vvs_bounded_inv_transform_beval)
{
    test_vec_x_logj_beval(vec_lower, scl_upper, vvs_transform, vvs_logj);
}

TEST_F(bounded_inv_transform_fixture, logj_vvv_bounded_inv_transform_feval)
{
    test_vec_x_logj_feval(vec_lower, vec_upper, vvv_transform, vvv_logj);
}

TEST_F(bounded_inv_transform_fixture, logj_vvv_bounded_inv_transform_beval)
{
    test_vec_x_logj_beval(vec_lower, vec_upper, vvv_transform, vvv_logj);
}

} // namespace boost
} // namespace ad
