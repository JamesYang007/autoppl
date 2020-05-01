#pragma once
#include "gtest/gtest.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <numeric>
#include <limits>

namespace ppl {

// Plotting utility to visualize histogram of samples.
template <class ArrayType>
inline void plot_hist(const ArrayType& arr,
                      double step_size = .5,
                      double min = std::numeric_limits<double>::lowest(),
                      double max = std::numeric_limits<double>::max()
                      )
{
    constexpr size_t nstars = 100;    // maximum number of stars to distribute

    min = (min == std::numeric_limits<double>::lowest()) ?
        *std::min_element(arr.begin(), arr.end()) :
        min;
    max = (max == std::numeric_limits<double>::max()) ?
        *std::max_element(arr.begin(), arr.end()) :
        max;
    const int64_t nearest_min = std::floor(min);
    const int64_t nearest_max = std::floor(max) + 1;
    const uint64_t range = nearest_max - nearest_min;
    const uint64_t n_hist = std::floor(range/step_size);

    // keeps count for each histogram bar
    std::vector<uint64_t> counter(n_hist, 0);

    for (auto x : arr) {
        if (nearest_min <= x && x <= nearest_max) {
            ++counter[std::floor((x - nearest_min) / step_size)];
        }
    }

    if ((min == *std::min_element(arr.begin(), arr.end())) &&
        (max == *std::max_element(arr.begin(), arr.end()))) {
        EXPECT_EQ(std::accumulate(counter.begin(), counter.end(), 0), (int) arr.size());
    }

    for (size_t i = 0; i < n_hist; ++i) {
        std::cout << i << "-" << (i+1) << ": " << '\t';
        std::cout << std::string(counter[i] * nstars/arr.size(), '*') << std::endl;
    }
}

inline void print_vector_stats(const std::vector<double>& arr, std::string name = "", int precision = 6)
{
	const double sum = std::accumulate(arr.cbegin(), arr.cend(), 0);
	double mean = sum / arr.size();

	auto stddev_fold = [mean](double base, double newItem) {
		return base + std::pow((newItem - mean), 2);
	};

	double sum_squares = std::accumulate(arr.begin(), arr.end(), 0.0, stddev_fold);
	double stddev = std::sqrt(sum_squares / arr.size() - 1);

	std::cout << name < " ";
	std::cout << std::scientific << std::setprecision(precision);
	std::cout << mean << " " << std::endl;
	std::cout << stddev << std::endl;
}


} // namespace ppl
