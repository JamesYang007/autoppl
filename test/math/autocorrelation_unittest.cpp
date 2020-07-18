#include "gtest/gtest.h"
#include <autoppl/math/autocorrelation.hpp>

namespace ppl {
namespace math {

struct autocorrelation_fixture : ::testing::Test
{
protected:

    static constexpr double tol = 1e-15;

    template <class T>
    auto brute_force(const arma::Mat<T>& x) 
    {
        arma::Mat<T> x_cent = x;
        arma::Mat<T> x_mean = arma::mean(x,0);
        x_cent.each_row([&](arma::rowvec& row) {
                    row -= x_mean;
                });

        arma::Mat<T> ac(arma::size(x), arma::fill::zeros);
        for (size_t j = 0; j < ac.n_cols; ++j) {
            auto col = ac.col(j);
            for (size_t i = 0; i < ac.n_rows; ++i) {
                for (size_t k = i; k < ac.n_rows; ++k) {
                    col(i) += x_cent(k,j) * x_cent(k-i,j);
                }
                col(i) /= (ac.n_rows * ac.n_rows * 2.);
            }
        }

        ac.each_col([](arma::vec& col) {
                    col /= col(0);
                });

        return ac;
    }

    template <class T>
    void check_results(const arma::Mat<T>& ac1,
                       const arma::Mat<T>& ac2)
    {
        for (size_t i = 0; i < ac1.n_rows; ++i) {
            for (size_t j = 0; j < ac1.n_cols; ++j) {
                EXPECT_NEAR(ac1(i,j), ac2(i,j), tol);
            }
        }
    }
};

TEST_F(autocorrelation_fixture, one_vec_three)
{
    arma::mat x(3,1,arma::fill::zeros);
    x(0,0) = 2; 
    x(1,0) = 3; 
    x(2,0) = -1;     

    arma::mat ac = autocorrelation(x);
    arma::mat ac_true = brute_force(x);
    
    check_results(ac, ac_true);
}

TEST_F(autocorrelation_fixture, two_vec_seven)
{
    arma::mat x(7,2,arma::fill::zeros);
    std::vector<double> x0({1.,-3.,2.,5.,1.,-0.32,0.32});
    std::vector<double> x1({8.9,0.1,-0.2,0.32,1.32,0.3,-0.001});

    for (size_t i = 0; i < x.n_rows; ++i) {
        x(i,0) = x0[i];
        x(i,1) = x1[i];
    }

    arma::mat ac = autocorrelation(x);
    arma::mat ac_true = brute_force(x);

    check_results(ac, ac_true);
}

} // namespace math
} // namespace ppl
