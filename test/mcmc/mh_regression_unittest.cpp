#include "gtest/gtest.h"
#include <vector>
#include <array>
#include <limits>
#include <autoppl/mcmc/mh/mh.hpp>
#include <autoppl/expression/program/program.hpp>
#include <autoppl/expression/model/bar_eq.hpp>
#include <autoppl/expression/model/glue.hpp>
#include <autoppl/expression/variable/data.hpp>
#include <autoppl/expression/variable/param.hpp>
#include <autoppl/expression/variable/binary.hpp>
#include <autoppl/expression/variable/constant.hpp>
#include <autoppl/expression/distribution/bernoulli.hpp>
#include <autoppl/expression/distribution/uniform.hpp>
#include <autoppl/expression/distribution/normal.hpp>
#include <autoppl/expression/op_overloads.hpp>
#include <testutil/sample_tools.hpp>

namespace ppl {

/*
 * Fixture for Metropolis-Hastings 
 */
struct mh_regression_fixture : ::testing::Test {
protected:
    using cont_value_t = util::cont_param_t;
    using p_cont_scl_t = Param<cont_value_t>;
    using d_cont_vec_t = Data<cont_value_t, ppl::vec>;

    size_t sample_size = 50000;
    size_t warmup = 1000;

    MHConfig config;

    cont_value_t tol = 1e-8;

    p_cont_scl_t w, b;
    d_cont_vec_t x, y, q, r;

    mh_regression_fixture()
        : x(6)
        , y(6)
        , q(6)
        , r(6)
    {
        x.get() << 2.5, 3, 3.5, 4, 4.5, 5.;
        y.get() << 3.5, 4, 4.5, 5, 5.5, 6.;
        q.get() << 2.4, 3.1, 3.6, 4, 4.5, 5.;
        r.get() << 3.5, 4, 4.4, 5.01, 5.46, 6.1;

        config.warmup = warmup;
        config.samples = sample_size;
    }

    template <class ArrayType>
    cont_value_t sample_average(const ArrayType& storage)
    {
        cont_value_t sum = std::accumulate(
                storage.data(), 
                storage.data() + storage.size(), 
                0.);
        return sum / storage.size();
    }
};

TEST_F(mh_regression_fixture, sample_regression_dist) {
    auto model = (w |= ppl::uniform(0., 2.),
                  b |= ppl::uniform(0., 2.),
                  y |= ppl::normal(x * w + b, 0.5)
    );

    auto out = ppl::mh(model, config);

    plot_hist(out.cont_samples.col(0), 0.2, 0., 1.);
    plot_hist(out.cont_samples.col(1), 0.2, 0., 1.);

    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 1.0, 0.1);
    EXPECT_NEAR(sample_average(out.cont_samples.col(1)), 1.0, 0.1);
}

TEST_F(mh_regression_fixture, sample_regression_fuzzy_dist) {
    auto model = (w |= ppl::uniform(0., 2.),
                  b |= ppl::uniform(0., 2.),
                  r |= ppl::normal(q * w + b, 0.5));

    auto out = ppl::mh(model, config);

    plot_hist(out.cont_samples.col(0), 0.2, 0., 1.);
    plot_hist(out.cont_samples.col(1), 0.2, 0., 1.);

    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 1.0, 0.1);
    EXPECT_NEAR(sample_average(out.cont_samples.col(1)), 0.95, 0.1);
}

TEST_F(mh_regression_fixture, sample_regression_normal_weight) {
    auto model = (w |= ppl::normal(0., 2.),
                  y |= ppl::normal(x * w + 1., 0.5));

    auto out = ppl::mh(model, config);

    plot_hist(out.cont_samples.col(0), 0.2, 0., 1.);

    EXPECT_NEAR(sample_average(out.cont_samples.col(0)), 1.0, 0.1);
}

} // namespace ppl
