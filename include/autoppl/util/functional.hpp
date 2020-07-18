#pragma once
#include <utility>

namespace ppl {
namespace util {

struct identity
{
    template <class... Ts, class T>
    constexpr T&& operator()(T&& x) const noexcept
    { return std::forward<T>(x); }
};

} // namespace util
} // namespace ppl
