#pragma once
#include "gtest/gtest.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <numeric>
#include <limits>
#include <fastad_bits/util/type_traits.hpp>

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

    double arr_min_elt;
    double arr_max_elt;

    if constexpr (ad::util::is_eigen_v<ArrayType>) {
        arr_min_elt = *std::min_element(arr.data(), arr.data() + arr.size());
        arr_max_elt = *std::max_element(arr.data(), arr.data() + arr.size());
    } else {
        arr_min_elt = *std::min_element(arr.begin(), arr.end());
        arr_max_elt = *std::max_element(arr.begin(), arr.end());
    }

    min = (min == std::numeric_limits<double>::lowest()) ?
        arr_min_elt : min;
    max = (max == std::numeric_limits<double>::max()) ?
        arr_max_elt : max;
    const int64_t nearest_min = std::floor(min);
    const int64_t nearest_max = std::floor(max) + 1;
    const uint64_t range = nearest_max - nearest_min;
    const uint64_t n_hist = std::floor(range/step_size);

    // keeps count for each histogram bar
    std::vector<uint64_t> counter(n_hist, 0);

    for (int i = 0; i < arr.size(); ++i) {
        if (nearest_min <= arr[i] && arr[i] <= nearest_max) {
            ++counter[std::floor((arr[i] - nearest_min) / step_size)];
        }
    }

    if ((min == arr_min_elt) &&
        (max == arr_max_elt)) {
        EXPECT_EQ(std::accumulate(counter.begin(), counter.end(), 0), (int) arr.size());
    }

    for (size_t i = 0; i < n_hist; ++i) {
        std::cout << i << "-" << (i+1) << ": " << '\t';
        std::cout << std::string(counter[i] * nstars/arr.size(), '*') << std::endl;
    }
}

} // namespace ppl
