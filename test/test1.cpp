#include "gtest/gtest.h"
#include <autoppl/dummy/fib.hpp>

namespace {

TEST(blaTest, test1) {
    int n = ppl::fib(10);
    EXPECT_EQ(n, 89);
}

}
