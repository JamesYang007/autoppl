#include <benchmark/benchmark.h>
#include "benchmark_utils.hpp"
#include <autoppl/autoppl.hpp>

namespace ppl {

static void BM_NormalWishart(benchmark::State& state) {
    size_t n_samples = state.range(0);

    ppl::Data<double, ppl::vec> y(2);
    y.get() << 1., -1.;
    Eigen::MatrixXd V(2,2);
    V << 2, 1, 1, 2;
    Param Sigma = make_param<double, mat>(2, pos_def());

    auto model = (Sigma |= wishart(V, 2.),
                  y |= normal(0, Sigma)
    );

    ppl::NUTSConfig<> config;
    config.samples = n_samples;
    config.warmup = n_samples;

    ppl::MCMCResult res;

	for (auto _ : state) {
		res = ppl::nuts(model, config);
	}

    ppl::summary("Sigma[0], Sigma[1], Sigma[2], Sigma[3]", 
                 res.cont_samples,
                 res.warmup_time, 
                 res.sampling_time);
}

BENCHMARK(BM_NormalWishart)->Arg(10000)
                           ->Arg(20000)
                           ->Arg(30000)
                           ->Arg(40000)
                           ->Arg(50000)
                            ;

} // namespace ppl
