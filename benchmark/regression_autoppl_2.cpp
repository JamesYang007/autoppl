#include <iostream>
#include <benchmark/benchmark.h>
#include "benchmark_utils.hpp"
#include <autoppl/autoppl.hpp>

namespace ppl {

static void BM_Regression(benchmark::State& state) {
    constexpr size_t n_data = 30000;
    size_t n_params = 3;

    Eigen::MatrixXd X_data(n_data, n_params);
    Eigen::VectorXd y_data(n_data);

    ppl::DataView<double, ppl::mat> X(X_data.data(), X_data.rows(), X_data.cols());
    ppl::DataView<double, ppl::vec> y(y_data.data(), X_data.rows());
    ppl::Param<double, ppl::vec> w(n_params);
    ppl::Param<double> b;

    std::mt19937 gen;
    std::normal_distribution n1(-1.0, 1.4);
    std::normal_distribution eps(0., 1.);

    Eigen::VectorXd beta_true(n_params);
    
    for (size_t j = 0; j < n_params; ++j) {
        beta_true(j) = static_cast<double>(j) / n_params;
    }

    for (size_t i = 0; i < n_data; ++i) {
        for (size_t j = 0; j < n_params; ++j) {
            X_data(i, j) = n1(gen);
        }
        y_data(i) = X_data.row(i) * beta_true + eps(gen);
    }

    auto model = (b |= ppl::normal(0., 5.),
                  w |= ppl::normal(0., 5.),
                  y |= ppl::normal(ppl::dot(X, w) + b, 1.0));

    MCMCResult res;

    for (auto _ : state) {
		res = ppl::nuts(model);
    }

    ppl::summary("b, w[0], w[1], ..., w[49]",
                 res.cont_samples,
                 res.warmup_time,
                 res.sampling_time);
}

BENCHMARK(BM_Regression);

} // namespace ppl
