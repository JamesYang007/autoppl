#include "gtest/gtest.h"
#include "autoppl.hpp"

namespace {

TEST(blaTest, test1) {
    int n = ppl::fib(10);
    EXPECT_EQ(n, 89);
}

}
