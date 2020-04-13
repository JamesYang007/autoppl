#include "gtest/gtest.h"
#include "autoppl.h"

TEST(blaTest, test1) {
    int n = fib(10);
    EXPECT_EQ(n, 89);
}