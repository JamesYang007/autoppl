#include <vector>
#include <benchmark/benchmark.h>
#include <autoppl/expression/variable/param.hpp>
#include <autoppl/mcmc/hmc/nuts/nuts.hpp>
#include <autoppl/expression/variable/data.hpp>
#include <autoppl/expression/expr_builder.hpp>
#include <autoppl/math/ess.hpp>

namespace ppl {

static void BM_NormalTwoPrior(benchmark::State& state) {
    size_t n_samples = state.range(0);
    constexpr size_t n_data = 1000;

	std::normal_distribution n(0.0, 1.0);
	std::mt19937 gen(0);
	ppl::Data<double, ppl::vec> y;

	ppl::Param<double> lambda1, lambda2, sigma;
	auto model = (
        sigma |= ppl::uniform(0.0, 20.0),
		lambda1 |= ppl::normal(0.0, 10.0),
		lambda2 |= ppl::normal(0.0, 10.0),
		y |= ppl::normal(lambda1 + lambda2, sigma)
	);

	for (size_t i = 0; i < n_data; ++i) {
		y.push_back(n(gen));
	}
	
    arma::mat storage(n_samples, 3);
	lambda1.storage() = storage.colptr(0);
	lambda2.storage() = storage.colptr(1);
    sigma.storage() = storage.colptr(2);

    ppl::NUTSConfig<> config;
    config.n_samples = n_samples;
    config.warmup = n_samples;

    ppl::NUTSResult res;

	for (auto _ : state) {
		res = ppl::nuts(model, config);
	}

    std::cout << "Warmup: " << res.warmup_time << std::endl;
    std::cout << "Sampling: " << res.sampling_time << std::endl;

    arma::mat mean = arma::mean(storage, 0);
    mean.print("Mean: l1, l2, s");

    arma::mat stddev = arma::stddev(storage, 0);
    stddev.print("Stddev: l1, l2, s");

    arma::mat ess = ppl::math::ess(storage);
    ess.print("ESS: l1, l2, s");

    arma::vec ess_per_s = ess / res.sampling_time;
    ess_per_s.print("ESS/s: w[0], w[1], w[2], b, s");
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
