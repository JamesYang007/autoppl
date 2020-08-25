#pragma once
#include <chrono>

namespace ppl {
namespace util {

template <class ClockType=std::chrono::steady_clock>
struct StopWatch
{
    void start() { start_ = ClockType::now(); }
    void stop() { end_ = ClockType::now(); }

    double elapsed() const
    {
        static constexpr double nano = 1e-9;
        return std::chrono::duration_cast<
        std::chrono::nanoseconds>(end_-start_).count() * nano;
    }

private:
    std::chrono::time_point<ClockType> start_;
    std::chrono::time_point<ClockType> end_;
};

} // namespace util
} // namespace ppl
