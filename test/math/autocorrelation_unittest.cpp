#include "gtest/gtest.h"
#include <autoppl/math/autocorrelation.hpp>

namespace ppl {
namespace math {

struct autocorrelation_fixture : ::testing::Test
{
protected:
    static constexpr double tol = 1e-15;

    template <class T>
    auto brute_force(const Eigen::MatrixBase<T>& x) 
    {
        using value_t = typename T::Scalar;
        using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;
        mat_t x_cent = x;
        x_cent.rowwise() -= x.colwise().mean();

        mat_t ac(x.rows(), x.cols());
        ac.setZero();

        for (int j = 0; j < ac.cols(); ++j) {
            auto col = ac.col(j);
            for (int i = 0; i < ac.rows(); ++i) {
                for (int k = i; k < ac.rows(); ++k) {
                    col(i) += x_cent(k,j) * x_cent(k-i,j);
                }
                col(i) /= (ac.rows() * ac.rows() * 2.);
            }
        }

        for (int j = 0; j < ac.cols(); ++j) {
            ac.col(j) /= ac.col(j)(0);
        }

        return ac;
    }

    template <class T>
    void check_results(const Eigen::MatrixBase<T>& ac1,
                       const Eigen::MatrixBase<T>& ac2)
    {
        for (int i = 0; i < ac1.rows(); ++i) {
            for (int j = 0; j < ac1.cols(); ++j) {
                EXPECT_NEAR(ac1(i,j), ac2(i,j), tol);
            }
        }
    }
};

TEST_F(autocorrelation_fixture, one_vec_three)
{
    Eigen::MatrixXd x(3,1);
    x(0,0) = 2; 
    x(1,0) = 3; 
    x(2,0) = -1;     

    Eigen::MatrixXd ac = autocorrelation(x);
    Eigen::MatrixXd ac_true = brute_force(x);
    
    check_results(ac, ac_true);
}

TEST_F(autocorrelation_fixture, two_vec_seven)
{
    Eigen::MatrixXd x(7,2);
    x.col(0) << 1.,-3.,2.,5.,1.,-0.32,0.32;
    x.col(1) << 8.9,0.1,-0.2,0.32,1.32,0.3,-0.001;

    Eigen::MatrixXd ac = autocorrelation(x);
    Eigen::MatrixXd ac_true = brute_force(x);

    check_results(ac, ac_true);
}

} // namespace math
} // namespace ppl
