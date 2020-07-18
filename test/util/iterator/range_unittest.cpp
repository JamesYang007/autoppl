#include "gtest/gtest.h"
#include <array>
#include <autoppl/util/iterator/range.hpp>

namespace ppl {
namespace util {

struct range_fixture : ::testing::Test
{
protected:
    static constexpr size_t size = 5;
    static constexpr int defval = 0;
    static constexpr size_t special_idx = 2;
    static constexpr int special_val = 10;

    using vector_t = std::vector<int>;
    using array_t = std::array<int, size>;
    using raw_array_t = int[size];
    vector_t v1;
    array_t v2;
    raw_array_t v3;
    range_fixture()
        : v1(size, defval)
        , v2{defval}
        , v3{defval}
    {
        v1[2] = special_val;
        v2[2] = special_val;
        v3[2] = special_val;
    }

    template <class Container>
    void test_size(const Container& c)
    {
        if constexpr (std::is_array_v<Container>) {
            auto r = range(c, c + size);
            EXPECT_EQ(r.size(), size);
        } else {
            auto r = range(c.begin(), c.end());
            EXPECT_EQ(r.size(), size);
        }
    }

    template <class Container>
    void test_op_paren(const Container& c)
    {
        if constexpr (std::is_array_v<Container>) {
            auto r = range(c, c + size);
            EXPECT_EQ(r(special_idx), special_val);
            for (size_t i = 0; i < size; ++i) {
                if (i != special_idx) { EXPECT_EQ(r(i), defval); }
            }
        } else {
            auto r = range(c.begin(), c.end());
            EXPECT_EQ(r(special_idx), special_val);
            for (size_t i = 0; i < size; ++i) {
                if (i != special_idx) { EXPECT_EQ(r(i), defval); }
            }
        }
    }
};

TEST_F(range_fixture, size)
{
    test_size(v1);
    test_size(v2);
    test_size(v3);
}

TEST_F(range_fixture, op_paren)
{
    test_op_paren(v1);
    test_op_paren(v2);
    test_op_paren(v3);
}

TEST_F(range_fixture, subrange)
{
    auto r = range(std::next(v1.begin(), 2), v1.end());
    EXPECT_EQ(r.size(), size - 2ul);
    EXPECT_EQ(r(special_idx - 2ul), special_val);
}
    
} // namespace util
} // namespace ppl
