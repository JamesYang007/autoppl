#include "gtest/gtest.h"
#include <autoppl/util/iterator/counting_iterator.hpp>

namespace ppl {
namespace util {

struct counting_iterator_fixture : ::testing::Test
{
protected:
    size_t val = 2;
    counting_iterator<size_t> it;
    counting_iterator_fixture()
        : it(val)
    {}
};

TEST_F(counting_iterator_fixture, op_star)
{
    EXPECT_EQ(*it, val);
}

TEST_F(counting_iterator_fixture, op_plus_plus)
{
    EXPECT_EQ(*(++it), val + 1);
    EXPECT_EQ(*it++, val + 1);
    EXPECT_EQ(*it, val + 2);
}

TEST_F(counting_iterator_fixture, op_minus_minus)
{
    EXPECT_EQ(*(--it), val - 1);
    EXPECT_EQ(*it--, val - 1);
    EXPECT_EQ(*it, val - 2);
}

TEST_F(counting_iterator_fixture, op_eq)
{
    EXPECT_EQ(counting_iterator<size_t>(val), it);
}

TEST_F(counting_iterator_fixture, op_neq)
{
    EXPECT_NE(counting_iterator<size_t>(0), it);
    EXPECT_NE(counting_iterator<size_t>(1), it);
    EXPECT_NE(counting_iterator<size_t>(3), it);
}
    
} // namespace util
} // namespace ppl
