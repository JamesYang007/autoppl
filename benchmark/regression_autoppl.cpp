#include <iostream>
#include <string>
#include <autoppl/expression/variable/data.hpp>
#include <autoppl/expression/variable/param.hpp>
#include <autoppl/expression/expr_builder.hpp>
#include <autoppl/mcmc/hmc/nuts/nuts.hpp>
#include <benchmark/benchmark.h>
#include <autoppl/math/ess.hpp>

namespace ppl {

static void BM_Regression(benchmark::State& state) {
    size_t num_samples = state.range(0);

    // load data
    std::string datapath = "life-clean.csv";
    arma::mat data;
    data.load(datapath);
    arma::mat X_data = data.tail_cols(data.n_cols-1);
    arma::vec y_data = data.col(0); // life expectancy

    // create data and param tags
    auto X = ppl::make_data_view<ppl::mat>(X_data);
    auto y = ppl::make_data_view<ppl::vec>(y_data);
    ppl::Param<double, ppl::vec> w(3);
    ppl::Param<double> b;
    ppl::Param<double> s;

    // create and bind sample storage
    arma::mat storage(num_samples, w.size() + b.size() + s.size());
    
    for (size_t i = 0; i < w.size(); ++i) {
        w.storage(i) = storage.colptr(i);
    }
    b.storage() = storage.colptr(w.size());
    s.storage() = storage.colptr(w.size() + b.size());
    
    // define model
    auto model = (s |= ppl::uniform(0.5, 8.),
                  b |= ppl::normal(0., 5.),
                  w |= ppl::normal(0., 5.),
                  y |= ppl::normal(ppl::dot(X, w) + b, s * s + 2.));
    
    // perform NUTS sampling
    NUTSConfig<> config;
    config.warmup = num_samples;
    config.n_samples = num_samples;

    NUTSResult res;

    for (auto _ : state) {
		res = ppl::nuts(model, config);
    }

    std::cout << "Warmup: " << res.warmup_time << std::endl;
    std::cout << "Sampling: " << res.sampling_time << std::endl;

    arma::mat mean = arma::mean(storage, 0);
    mean.print("Mean: w[0], w[1], w[2], b, s");

    arma::mat stddev = arma::stddev(storage, 0);
    stddev.print("Mean: w[0], w[1], w[2], b, s");

    arma::vec ess = math::ess(storage);
    ess.print("ESS: w[0], w[1], w[2], b, s");

    arma::vec ess_per_s = ess / res.sampling_time;
    ess_per_s.print("ESS/s: w[0], w[1], w[2], b, s");

}

BENCHMARK(BM_Regression)->Arg(100)
                        ->Arg(500)
                        ->Arg(1000)
                        ->Arg(5000)
                        ->Arg(10000)
                        ->Arg(50000)
                        ->Arg(100000);

} // namespace ppl
