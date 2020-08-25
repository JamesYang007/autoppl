#include <random>
#include <benchmark/benchmark.h>
#include "benchmark_utils.hpp"
#include <autoppl/autoppl.hpp>

namespace ppl {

static void BM_MHBayesNet(benchmark::State& state) {
    size_t n_samples = state.range(0);
    constexpr size_t n_data = 1000;

	std::bernoulli_distribution b(0.341);
	std::mt19937 gen(0);
	ppl::Data<util::disc_param_t, ppl::vec> y(n_data);

	ppl::Param m1 = ppl::make_param<double>(ppl::bounded(0., 1.)); 
	ppl::Param m2 = ppl::make_param<double>(ppl::bounded(0., 1.)); 
	ppl::Param M1 = ppl::make_param<double>(ppl::bounded(m1, 1.)); 
	ppl::Param M2 = ppl::make_param<double>(ppl::bounded(m2, 1.)); 
	ppl::Param p1 = ppl::make_param<double>(ppl::bounded(m1, M1)); 
	ppl::Param p2 = ppl::make_param<double>(ppl::bounded(m2, M2)); 
    ppl::Param<int> w;
	auto model = (
        m1 |= ppl::uniform(0., 1.),
        m2 |= ppl::uniform(0., 1.),
        M1 |= ppl::uniform(0., 1.),
        M2 |= ppl::uniform(0., 1.),
        p1 |= ppl::uniform(m1, M1),
        p2 |= ppl::uniform(m2, M2),
        w |= ppl::bernoulli(0.3 * p1),
        y |= ppl::bernoulli(w * p1 + (1-w) * p2)
    );

	for (size_t i = 0; i < n_data; ++i) {
		y.get()(i) = b(gen);
	}

    ppl::MCMCResult res;

    ppl::MHConfig config;
    config.warmup = n_samples;
    config.samples = n_samples;
	
	for (auto _ : state) {
		res = ppl::mh(model, config);
	}

    ppl::summary("m1, m2, M1, M2, p1, p2", 
                 res.cont_samples,
                 res.warmup_time,
                 res.sampling_time);
    ppl::summary("w", 
                 res.disc_samples.cast<double>(),
                 res.warmup_time, 
                 res.sampling_time);
}

BENCHMARK(BM_MHBayesNet)
    ->Arg(100000)
    ->Arg(200000)
    ->Arg(300000)
    ;
} // namespace ppl
