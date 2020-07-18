#include "gtest/gtest.h"
#include <autoppl/math/ess.hpp>

namespace ppl {
namespace math {

struct ess_fixture : ::testing::Test
{
protected:
};

TEST_F(ess_fixture, test)
{
    arma::cube samples(6,2,3);

    samples(0,0,0) = 2;
    samples(1,0,0) = 3;
    samples(2,0,0) = -1;
    samples(3,0,0) = 2;
    samples(4,0,0) = 5;
    samples(5,0,0) = 10;

    samples(0,1,0) = 1;
    samples(1,1,0) = 2;
    samples(2,1,0) = 3;
    samples(3,1,0) = 4;
    samples(4,1,0) = 5;
    samples(5,1,0) = 6;

    samples(0,0,1) = 1;
    samples(1,0,1) = -1;
    samples(2,0,1) = 2;
    samples(3,0,1) = 5;
    samples(4,0,1) = 4;
    samples(5,0,1) = 7;

    samples(0,1,1) = 2;
    samples(1,1,1) = -1;
    samples(2,1,1) = 1;
    samples(3,1,1) = 4;
    samples(4,1,1) = 2;
    samples(5,1,1) = -2;

    samples(0,0,2) = 1;
    samples(1,0,2) = -1;
    samples(2,0,2) = 0;
    samples(3,0,2) = 0;
    samples(4,0,2) = 3;
    samples(5,0,2) = -2;

    samples(0,1,2) = 0;
    samples(1,1,2) = -1;
    samples(2,1,2) = 1;
    samples(3,1,2) = 4;
    samples(4,1,2) = 2;
    samples(5,1,2) = -2;

    arma::vec ESS = ess(samples);
    ESS.print("ESS");
}

} // namespace math
} // namespace ppl
