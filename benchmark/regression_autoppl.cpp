#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <numeric>

#include <autoppl/variable.hpp>
#include <autoppl/expr_builder.hpp>
#include <autoppl/algorithm/mh.hpp>
#include <autoppl/algorithm/nuts.hpp>

#include <benchmark/benchmark.h>

namespace ppl {

template <class ArrayType>
double sample_average(const ArrayType& storage) {
    double sum = std::accumulate(
        storage.begin(),
        storage.end(),
        0.);
    return sum / (storage.size());
}

static void BM_Regression(benchmark::State& state) {
    constexpr size_t num_samples = 1000;

    std::array<std::string, 4> headers = {"Life expectancy", "Alcohol", "HIV/AIDS", "GDP"};

    std::unordered_map<std::string, ppl::Data<double>> data;
    std::unordered_map<std::string, ppl::Param<double>> params;
    std::array<std::vector<double>, 4> storage;

    // Read in data
    std::fstream fin;
    fin.open("life-clean.csv", std::ios::in);
    std::string line;
    char temp;
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

    for (auto _ : state) {
		ppl::nuts(model, 1000, num_samples, 1000, 1, 10, 0.95);
    }

	std::cout << "Alcohol w: " << sample_average(storage[1]) << std::endl;
	std::cout << "HIV/AIDS w: " << sample_average(storage[2]) << std::endl;
	std::cout << "GDP: " << sample_average(storage[3]) << std::endl;
	std::cout << "Bias: " << sample_average(storage[0]) << std::endl;
}

BENCHMARK(BM_Regression);

} // namespace ppl
