#include <benchmark/benchmark.h>
#include "benchmark_utils.hpp"
#include <autoppl/autoppl.hpp>

namespace ppl {

template <class VecType>
void generate_data(VecType& y, double phi, double sigma, double mu)
{
	std::normal_distribution n(0.0, 1.0);
	std::mt19937 gen(ppl::mcmc::random_seed());

    double h = n(gen) * sigma / 
                std::sqrt(1. - phi * phi) 
                + mu;
    y[0] = n(gen) * std::exp(h/2.);

    for (int i = 1; i < y.size(); ++i) {
        h = n(gen) * sigma + mu + phi * (h - mu);
        y[i] = n(gen) * std::exp(h/2.);
    }
}

static void BM_StochasticVolatility(benchmark::State& state) {
    size_t n_samples = state.range(0);
    constexpr size_t n_data = 500;

    double actual_phi = 0.95;
    double actual_sigma = 0.25;
    double actual_mu = -1.02;
    Eigen::VectorXd y_data(n_data);
    generate_data(y_data, actual_phi, actual_sigma, actual_mu);

    ppl::DataView<double, ppl::vec> y(y_data.data(), n_data);
    ppl::Param phi = ppl::make_param<double>(ppl::bounded(-1., 1.));
    ppl::Param sigma = ppl::make_param<double>(ppl::lower(0.));
    ppl::Param<double> mu;
    ppl::Param<double, ppl::vec> h_std(n_data);
    ppl::TParam<double, ppl::vec> h(n_data);

    auto tp_expr = (
        h = h_std * sigma,
        h[0] /= ppl::sqrt(1. - phi * phi),
        h += mu,
        ppl::for_each(ppl::util::counting_iterator<>(1),
                      ppl::util::counting_iterator<>(h.size()),
                      [&](size_t i) { return h[i] += phi * (h[i-1] - mu); })
    );

    auto model = (
        phi |= ppl::uniform(-1., 1.),
        sigma |= ppl::cauchy(0., 5.),
        mu |= ppl::cauchy(0., 10.),
        h_std |= ppl::normal(0., 1.),
        y |= ppl::normal(0., ppl::exp(h / 2.))
    );

    auto program = tp_expr | model;

    ppl::NUTSConfig<> config;
    config.samples = n_samples;
    config.warmup = n_samples;

    ppl::MCMCResult res;

	for (auto _ : state) {
		res = ppl::nuts(program, config);
	}

    Eigen::MatrixXd sub_samples = res.cont_samples.block(0,0,n_samples,3);
    ppl::summary("phi, sigma, mu", sub_samples,
                 res.warmup_time, res.sampling_time);
}

BENCHMARK(BM_StochasticVolatility)
    //->Arg(100)
    //->Arg(500)
    //->Arg(1000)
    //->Arg(3000)
    ->Arg(5000)
    //->Arg(7000)
    //->Arg(10000)
    //->Arg(20000)
    ;
} // namespace ppl
