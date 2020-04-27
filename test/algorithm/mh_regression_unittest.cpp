#include "gtest/gtest.h"
#include <array>
#include <limits>
#include <autoppl/algorithm/mh.hpp>
#include <autoppl/expr_builder.hpp>
#include <testutil/sample_tools.hpp>
#include <vector>
#include <iostream>

namespace ppl {

/*
 * Fixture for Metropolis-Hastings 
 */
struct mh_fixture : ::testing::Test {
protected:
    size_t sample_size = 20000;
    double tol = 1e-8;

    std::vector<double> w_storage, b_storage;
    Param<double> w, b;

    ppl::Data<double> x {2.5, 3, 3.5, 4, 4.5, 5.};
    ppl::Data<double> y {3.5, 4, 4.5, 5, 5.5, 6.};

    size_t burn = 1000;

    mh_fixture()
        : w_storage(sample_size)
        , b_storage(sample_size)
        , w{w_storage.data()}
        , b{b_storage.data()}
    {}

    template <class ArrayType>
    double sample_average(const ArrayType& storage)
    {
        double sum = std::accumulate(
                std::next(storage.begin(), burn), 
                storage.end(), 
                0.);
        return sum / (storage.size() - burn);
    }
};

TEST_F(mh_fixture, test_regression_pdf) {
    w.set_value(1.0);
    b.set_value(1.0);

    std::cout << x.size() << std::endl;
    std::cout << (x * w + b).size() << " " << x.size() << " " << w.size() << " " << b.size() << std::endl;
    std::cout << (x * w + b).get_value(0) << std::endl;

    EXPECT_EQ((x * w + b).get_value(0), 3.5);

    auto model = (w |= ppl::uniform(0, 2),
                  b |= ppl::uniform(0, 2),
                  y |= ppl::normal(x * w + b, 0.5));
    
    EXPECT_NEAR(model.pdf(), 0.24177490849077804, tol);
    EXPECT_NEAR(model.log_pdf(), -2.806042476988255, tol);
}

TEST_F(mh_fixture, sample_regression_dist) {
    auto model = (w |= ppl::uniform(0, 2),
                  b |= ppl::uniform(0, 2),
                  y |= ppl::normal(x * w + b, 0.5)
    );

    ppl::mh_posterior(model, sample_size);

    plot_hist(w_storage, 0.2, 0., 1.);
    plot_hist(b_storage, 0.2, 0., 1.);

    EXPECT_NEAR(sample_average(w_storage), 1.0, 0.1);
    EXPECT_NEAR(sample_average(b_storage), 1.0, 0.1);
}

} // ppl