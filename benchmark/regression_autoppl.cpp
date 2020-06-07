#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <unordered_map>

#include <autoppl/variable.hpp>
#include <autoppl/expr_builder.hpp>
#include <autoppl/mcmc/hmc/nuts/nuts.hpp>

#include "benchmark_utils.hpp"

#include <benchmark/benchmark.h>

namespace ppl {

template <class ArrayType>
inline double stddev(const ArrayType& v)
{
    double mean = std::accumulate(v.begin(), v.end(), 0.)/v.size();
    double var = 0.;
    for (auto x : v) {
        auto diff = (x - mean);
        var += diff * diff;
    }
    return std::sqrt(var/(v.size()));
}

static void BM_Regression(benchmark::State& state) {
    size_t num_samples = state.range(0);

    std::array<std::string, 4> headers = {"Life expectancy", "Alcohol", "HIV/AIDS", "GDP"};

    std::unordered_map<std::string, ppl::Data<double>> data;
    std::unordered_map<std::string, ppl::Param<double>> params;
    std::array<std::vector<double>, 4> storage;

    // Read in data
    std::fstream fin;
    fin.open("life-clean.csv", std::ios::in);
    std::string line;
    double value;
    while (std::getline(fin, line, '\n')) { 
        auto it = headers.begin();
        std::stringstream s(line);
        while (s >> value) {
            data[*it].observe(value);
            ++it;
        }
    }

    // resize each storage and bind with param
    int i = 0;
    for (auto it = headers.begin(); it != headers.end(); ++it, ++i) {
        storage[i].resize(num_samples);
        params[*it].set_storage(storage[i].data());
    }

    auto model = (params["Alcohol"] |= ppl::normal(0., 5.),
                  params["HIV/AIDS"] |= ppl::normal(0., 5.),
                  params["GDP"] |= ppl::normal(0., 5.),
                  params["Life expectancy"] |= ppl::normal(0., 5.),

                  data["Life expectancy"] |= ppl::normal(
                    params["Alcohol"] * data["Alcohol"] +
                    params["HIV/AIDS"] * data["HIV/AIDS"] +
                    params["GDP"] * data["GDP"] + params["Life expectancy"], 5.0));
    
    NUTSConfig<> config = {
        .warmup = num_samples,
        .n_samples = num_samples
    };
    for (auto _ : state) {
		ppl::nuts(model, config);
    }

	std::cout << "Bias: " << sample_average(storage[0]) << std::endl;
	std::cout << "Alcohol w: " << sample_average(storage[1]) << std::endl;
	std::cout << "HIV/AIDS w: " << sample_average(storage[2]) << std::endl;
	std::cout << "GDP: " << sample_average(storage[3]) << std::endl;

	std::cout << "Bias: " << stddev(storage[0]) << std::endl;
	std::cout << "Alcohol w: " << stddev(storage[1]) << std::endl;
	std::cout << "HIV/AIDS w: " << stddev(storage[2]) << std::endl;
	std::cout << "GDP: " << stddev(storage[3]) << std::endl;
}

BENCHMARK(BM_Regression)->Arg(100)->Arg(500)->Arg(1000)->Arg(5000)->Arg(10000)->Arg(50000)->Arg(100000);

} // namespace ppl
