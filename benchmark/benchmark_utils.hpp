#pragma once
#include <numeric>

namespace ppl {

template <class ArrayType>
double sample_average(const ArrayType& storage) {
    double sum = std::accumulate(
        storage.begin(),
        storage.end(),
        0.);
    return sum / (storage.size());
}

} // namespace ppl
