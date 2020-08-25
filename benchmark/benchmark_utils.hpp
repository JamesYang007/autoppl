#pragma once
#include <numeric>
#include <iostream>
#include <Eigen/Dense>
#include <autoppl/math/ess.hpp>

namespace ppl {

template <class Derived>
inline auto mean(const Eigen::MatrixBase<Derived>& m)
{ return m.colwise().mean(); }

template <class Derived>
inline auto sd(const Eigen::MatrixBase<Derived>& m)
{
    assert(m.rows() > 1);
    auto var = (m.rowwise() - ppl::mean(m))
                    .colwise().squaredNorm() / (m.rows() - 1);
    return var.array().sqrt().matrix();
}

inline void summary(const std::string& header,
                    const Eigen::MatrixXd& m,
                    double warmup_time,
                    double sampling_time)
{
    std::cout << "Warmup: " << warmup_time << std::endl;
    std::cout << "Sampling: " << sampling_time << std::endl;

    std::cout << header << std::endl;

    Eigen::MatrixXd mean = ppl::mean(m);
    std::cout << "Mean:\n"
              << mean << std::endl; 

    Eigen::MatrixXd sd = ppl::sd(m);
    std::cout << "SD:\n"
              << sd << std::endl; 

    Eigen::MatrixXd ess = ppl::math::ess(m);
    std::cout << "ESS:\n"
              << ess << std::endl;

    Eigen::MatrixXd ess_per_s = ess / sampling_time;
    std::cout << "ESS/s:\n"
              << ess_per_s << std::endl;
}

} // namespace ppl
