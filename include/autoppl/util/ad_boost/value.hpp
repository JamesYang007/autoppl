#pragma once
#include <fastad_bits/util/type_traits.hpp>

namespace ad {
namespace boost {

template <class T>
inline auto sum(const T& x)
{
    if constexpr (ad::util::is_eigen_v<T>) {
        return x.sum();
    } else {
        return x;
    }
}

} // namespace boost
} // namespace ad
