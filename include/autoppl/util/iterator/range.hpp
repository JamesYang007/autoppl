#pragma once
#include <cassert>
#include <functional>

namespace ppl {
namespace util {

/**
 * Small class to view a range of elements. 
 */
template <class Iter>
struct range
{
    using iter_t = Iter;

    range(iter_t begin, iter_t end)
        : begin_{begin}
        , end_{end}
        , size_{static_cast<size_t>(std::distance(begin, end))}
    {}

    auto& operator()(size_t i) { 
        assert(i < size_);
        return *std::next(begin_, i); 
    }

    const auto& operator()(size_t i) const { 
        assert(i < size_);
        return *std::next(begin_, i); 
    }

    iter_t begin() { return begin_; }
    const iter_t begin() const { return begin_; }

    iter_t end() { return end_; }
    const iter_t end() const { return end_; }

    size_t size() const { return size_; }

    void bind(iter_t begin, iter_t end)
    {
        begin_ = begin;
        end_ = end;
        size_ = static_cast<size_t>(std::distance(begin, end));
    }

private:
    iter_t begin_;
    iter_t end_;
    size_t size_;
};

} // namespace util
} // namespace ppl
