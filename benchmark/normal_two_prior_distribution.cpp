#include <chrono>
#include <benchmark/benchmark.h>
#include <autoppl/autoppl.hpp>
#include "benchmark_utils.hpp"

namespace ppl {

static void BM_NormalTwoPrior(benchmark::State& state) {
    constexpr size_t n_samples = 1000;
    constexpr size_t n_data = 1000;

	std::normal_distribution n(0.0, 1.0);
	std::mt19937 gen(0);
	ppl::Data<double> y;

	ppl::Param<double> lambda1, lambda2, sigma;
	auto model = (
        sigma |= ppl::uniform(0.0, 20.0),
		lambda1 |= ppl::normal(0.0, 10.0),
		lambda2 |= ppl::normal(0.0, 10.0),
		y |= ppl::normal(lambda1 + lambda2, sigma)
	);

	for (size_t i = 0; i < n_data; ++i) {
		y.observe(n(gen));
	}
	
	std::array<double, n_samples> l1_storage, l2_storage, s_storage;
	lambda1.set_storage(l1_storage.data());
	lambda2.set_storage(l2_storage.data());
    sigma.set_storage(s_storage.data());

	for (auto _ : state) {
		ppl::nuts(model);
	}

    std::cout << "l1: " << sample_average(l1_storage) << std::endl;
    std::cout << "l2: " << sample_average(l2_storage) << std::endl;
    std::cout << "s: " << sample_average(s_storage) << std::endl;
}

BENCHMARK(BM_NormalTwoPrior);

} // namespace ppl
