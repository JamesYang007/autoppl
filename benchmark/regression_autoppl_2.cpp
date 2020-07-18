#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <unordered_map>

#include <autoppl/expression/variable/data.hpp>
#include <autoppl/expression/variable/param.hpp>
#include <autoppl/expression/expr_builder.hpp>
#include <autoppl/mcmc/hmc/nuts/nuts.hpp>

#include "benchmark_utils.hpp"

#include <benchmark/benchmark.h>

namespace ppl {

static void BM_Regression(benchmark::State& state) {
    constexpr size_t num_samples = 1000;
    constexpr size_t n_data = 30000;

    std::array<std::string, 4> headers = {"b", "x1", "x2", "x3"};

    std::unordered_map<std::string, ppl::Data<double, ppl::vec>> data;
    std::unordered_map<std::string, ppl::Param<double>> params;
    std::array<std::vector<double>, 4> storage;

    std::mt19937 gen;
    std::normal_distribution n1(-1.0, 1.4);
    std::normal_distribution n2(0.0, 1.4);
    std::normal_distribution n3(1.0, 1.4);
    std::normal_distribution eps(0.0, 1.0);

    for (size_t i = 0; i < n_data; ++i) {
        double x1 = n1(gen);
        double x2 = n2(gen);
        double x3 = n3(gen);
        data[headers[1]].push_back(x1);
        data[headers[2]].push_back(x2);
        data[headers[3]].push_back(x3);
        data["y"].push_back(x1 * 1.4 + x2 * 2. + x3 * 0.32 + eps(gen));
    }

    // resize each storage and bind with param
    int i = 0;
    for (auto it = headers.begin(); it != headers.end(); ++it, ++i) {
        storage[i].resize(num_samples);
        params[*it].storage() = storage[i].data();
    }

    auto model = (params["b"] |= ppl::normal(0., 5.),
                  params["x1"] |= ppl::normal(0., 5.),
                  params["x2"] |= ppl::normal(0., 5.),
                  params["x3"] |= ppl::normal(0., 5.),

                  data["y"] |= ppl::normal(
                    params["x1"] * data["x1"] +
                    params["x2"] * data["x2"] +
                    params["x3"] * data["x3"] + 
                    params["b"], 1.0));

    for (auto _ : state) {
		ppl::nuts(model);
    }

	std::cout << "b:  " << sample_average(storage[0]) << std::endl;
	std::cout << "w1: " << sample_average(storage[1]) << std::endl;
	std::cout << "w2: " << sample_average(storage[2]) << std::endl;
	std::cout << "w3: " << sample_average(storage[3]) << std::endl;
}

BENCHMARK(BM_Regression);

} // namespace ppl
