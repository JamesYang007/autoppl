#include <benchmark/benchmark.h>
#include "benchmark_utils.hpp"
#include <autoppl/autoppl.hpp>

namespace ppl {

static void BM_NormalTwoPrior(benchmark::State& state) {
    size_t n_samples = state.range(0);
    constexpr size_t n_data = 1000;

	std::normal_distribution n(0.0, 1.0);
	std::mt19937 gen(0);
	ppl::Data<double, ppl::vec> y(n_data);

	ppl::Param<double> lambda1, lambda2, sigma;
	auto model = (
        sigma |= ppl::uniform(0.0, 20.0),
		lambda1 |= ppl::normal(0.0, 10.0),
		lambda2 |= ppl::normal(0.0, 10.0),
		y |= ppl::normal(lambda1 + lambda2, sigma)
	);

	for (size_t i = 0; i < n_data; ++i) {
		y.get()(i) = n(gen);
	}
	
    ppl::NUTSConfig<> config;
    config.samples = n_samples;
    config.warmup = n_samples;

    ppl::MCMCResult res;

	for (auto _ : state) {
		res = ppl::nuts(model, config);
	}

    ppl::summary("s, l1, l2", res.cont_samples,
                 res.warmup_time, res.sampling_time);
}

BENCHMARK(BM_NormalTwoPrior)->Arg(100)
                            ->Arg(500)
                            ->Arg(1000)
                            ->Arg(3000)
                            ->Arg(5000)
                            ->Arg(7000)
                            ->Arg(10000)
                            ->Arg(20000);
} // namespace ppl
