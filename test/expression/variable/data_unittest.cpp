#include <autoppl/variable.hpp>

#include "gtest/gtest.h"

namespace ppl {
namespace expr {

struct data_fixture : ::testing::Test {
   protected:
    Data<double> var1 {1.0, 2.0, 3.0};
    Data<double> var2 {1.0};

    size_t expected_size;
    size_t real_size;
};

TEST_F(data_fixture, test_multiple_value) {
    expected_size = 3;
    real_size = var1.size();
    
    EXPECT_EQ(expected_size, real_size);

    expected_size = 1;
    real_size = var2.size();
    
    EXPECT_EQ(expected_size, real_size);

    EXPECT_EQ(var1.get_value(0), 1.0);
    EXPECT_EQ(var1.get_value(1), 2.0);
    EXPECT_EQ(var1.get_value(2), 3.0);

    EXPECT_DEATH({
        var2.get_value(1);
    }, "");

    EXPECT_DEATH({
        var2.get_value(-1);
    }, "");

    EXPECT_DEATH({
        var1.get_value(3);
    }, "");

    var1.clear();
    expected_size = 0;
    real_size = var1.size();
    EXPECT_EQ(expected_size, real_size);

    var1.observe(0.1);
    var1.observe(0.2);

    expected_size = 2;
    real_size = var1.size();
    EXPECT_EQ(expected_size, real_size);
}

}  // namespace expr
}  // namespace ppl
