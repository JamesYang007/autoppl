#include <benchmark/benchmark.h>
#include <autoppl/autoppl.hpp>
#include <numeric>

namespace ppl {

static void BM_NormalTwoPrior(benchmark::State& state) {
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

	for (int i = 0; i < 100; i++) {
		y.observe(n(gen));
	}
	
	std::array<double, 1000> l1_storage, l2_storage, s_storage;
	lambda1.set_storage(l1_storage.data());
	lambda2.set_storage(l2_storage.data());
    sigma.set_storage(s_storage.data());

	for (auto _ : state) {
		ppl::nuts(model, 1000, 1000, 1000);
	}
}

BENCHMARK(BM_NormalTwoPrior);

} // namespace ppl
