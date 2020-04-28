#include <autoppl/variable.hpp>

#include "gtest/gtest.h"

namespace ppl {
namespace expr {

struct param_fixture : ::testing::Test {
   protected:
    Param<double> param1;
    Param<double> param2 {3.};

    size_t expected_size;
    size_t real_size;
};

TEST_F(param_fixture, test_multiple_value) {
    expected_size = 1;
    real_size = param1.size();
    
    EXPECT_EQ(expected_size, real_size);

    EXPECT_EQ(param1.get_value(0), 0.0);
    param1.set_value(1.0);

    EXPECT_EQ(param1.get_value(0), 1.0);
    EXPECT_EQ(param1.get_value(10), 1.0);  // all indices return the same

    EXPECT_EQ(param2.get_value(0), 3.0);  // all indices return the same

    EXPECT_EQ(param1.get_storage(), nullptr);
    
    double storage[5];
    param1.set_storage(storage);
    EXPECT_EQ(param1.get_storage(), storage);
}

}  // namespace expr
}  // namespace ppl
