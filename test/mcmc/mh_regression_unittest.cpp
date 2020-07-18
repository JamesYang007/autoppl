#include "gtest/gtest.h"
#include <array>
#include <limits>
#include <autoppl/mcmc/mh.hpp>
#include <autoppl/expression/expr_builder.hpp>
#include <testutil/sample_tools.hpp>
#include <vector>

namespace ppl {

/*
 * Fixture for Metropolis-Hastings 
 */
struct mh_regression_fixture : ::testing::Test {
protected:
    using cont_value_t = double;
    using p_cont_scl_t = Param<cont_value_t>;
    using d_cont_vec_t = Data<cont_value_t, ppl::vec>;

    size_t sample_size = 50000;
    cont_value_t tol = 1e-8;

    std::vector<cont_value_t> w_storage, b_storage;
    p_cont_scl_t w, b;

    d_cont_vec_t x {2.5, 3, 3.5, 4, 4.5, 5.};
    d_cont_vec_t y {3.5, 4, 4.5, 5, 5.5, 6.};

    d_cont_vec_t q{2.4, 3.1, 3.6, 4, 4.5, 5.};
    d_cont_vec_t r{3.5, 4, 4.4, 5.01, 5.46, 6.1};

    size_t warmup = 1000;

    mh_regression_fixture()
        : w_storage(sample_size)
        , b_storage(sample_size)
        , w{w_storage.data()}
        , b{b_storage.data()}
    {}

    template <class ArrayType>
    cont_value_t sample_average(const ArrayType& storage)
    {
        cont_value_t sum = std::accumulate(
                std::next(storage.begin(), warmup), 
                storage.end(), 
                0.);
        return sum / (storage.size() - warmup);
    }
};

TEST_F(mh_regression_fixture, sample_regression_dist) {
    auto model = (w |= ppl::uniform(0., 2.),
                  b |= ppl::uniform(0., 2.),
                  y |= ppl::normal(x * w + b, 0.5)
    );

    ppl::mh(model, sample_size);

    plot_hist(w_storage, 0.2, 0., 1.);
    plot_hist(b_storage, 0.2, 0., 1.);

    EXPECT_NEAR(sample_average(w_storage), 1.0, 0.1);
    EXPECT_NEAR(sample_average(b_storage), 1.0, 0.1);
}

TEST_F(mh_regression_fixture, sample_regression_fuzzy_dist) {
    auto model = (w |= ppl::uniform(0., 2.),
                  b |= ppl::uniform(0., 2.),
                  r |= ppl::normal(q * w + b, 0.5));

    ppl::mh(model, sample_size);

    plot_hist(w_storage, 0.2, 0., 1.);
    plot_hist(b_storage, 0.2, 0., 1.);

    EXPECT_NEAR(sample_average(w_storage), 1.0, 0.1);
    EXPECT_NEAR(sample_average(b_storage), 0.95, 0.1);
}

TEST_F(mh_regression_fixture, sample_regression_normal_weight) {
    auto model = (w |= ppl::normal(0., 2.),
                  y |= ppl::normal(x * w + 1., 0.5));

    ppl::mh(model, sample_size);

    plot_hist(w_storage, 0.2, 0., 1.);

    EXPECT_NEAR(sample_average(w_storage), 1.0, 0.1);
}

} // namespace ppl
