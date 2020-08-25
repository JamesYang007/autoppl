#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <benchmark/benchmark.h>
#include "benchmark_utils.hpp"
#include <autoppl/autoppl.hpp>

namespace ppl {

void load_data(const std::string& path,
               Eigen::MatrixXd& m,
               char delim = ' ')
{
    std::vector<double> row;
    std::ifstream stream(path);
    std::string row_str;
    std::string entry_str;

    int rows = 0;
    while (std::getline(stream, row_str)) {
        std::stringstream sstream(row_str);
        while (getline(sstream, entry_str, delim)) {
            row.push_back(std::stod(entry_str));
        }
        ++rows;
    }

    using map_t = Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >;
    m = map_t(row.data(), rows, row.size() / rows);
}

static void BM_Regression(benchmark::State& state) {
    size_t num_samples = state.range(0);

    // load data
    Eigen::MatrixXd data;
    load_data("life-clean.csv", data);
    auto X_data = data.block(0, 1, data.rows(), data.cols()-1);
    auto y_data = data.col(0);

    // create data and param tags
    DataView<double, mat> X(X_data.data(), X_data.rows(), X_data.cols());
    DataView<double, vec> y(y_data.data(), y_data.rows());
    ppl::Param<double, ppl::vec> w(3);
    ppl::Param<double> b;
    ppl::Param<double> s;
    
    // define model
    auto model = (s |= ppl::uniform(0.5, 8.),
                  b |= ppl::normal(0., 5.),
                  w |= ppl::normal(0., 5.),
                  y |= ppl::normal(ppl::dot(X, w) + b, s * s + 2.));
    
    // perform NUTS sampling
    NUTSConfig<> config;
    config.warmup = num_samples;
    config.samples = num_samples;

    MCMCResult res;

    for (auto _ : state) {
		res = ppl::nuts(model, config);
    }

    ppl::summary("s, b, w[0], w[1], w[2]", res.cont_samples,
                 res.warmup_time, res.sampling_time);
}

BENCHMARK(BM_Regression)->Arg(100)
                        ->Arg(500)
                        ->Arg(1000)
                        ->Arg(5000)
                        ->Arg(10000)
                        ->Arg(50000)
                        ->Arg(100000);

} // namespace ppl
