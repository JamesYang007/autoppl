#include "gtest/gtest.h"
#include <autoppl/math/ess.hpp>

namespace ppl {
namespace math {

struct ess_fixture : ::testing::Test
{
protected:
    std::vector<Eigen::MatrixXd> v;
    std::vector<std::reference_wrapper<const Eigen::MatrixXd>> v_ref;
    ess_fixture()
        : v(3, {6,2})
        , v_ref()
    {
        for (auto& m : v) { v_ref.emplace_back(m); }
    }
};

TEST_F(ess_fixture, test)
{
    v[0](0,0) = 2;
    v[0](1,0) = 3;
    v[0](2,0) = 5;
    v[0](3,0) = 6;
    v[0](4,0) = 9;
    v[0](5,0) = 10;

    v[0](0,1) = 1;
    v[0](1,1) = 2;
    v[0](2,1) = 3;
    v[0](3,1) = 4;
    v[0](4,1) = 5;
    v[0](5,1) = 6;

    v[1](0,0) = 1;
    v[1](1,0) = 1;
    v[1](2,0) = 2;
    v[1](3,0) = 5;
    v[1](4,0) = 4;
    v[1](5,0) = 7;

    v[1](0,1) = 2;
    v[1](1,1) = 5;
    v[1](2,1) = 6;
    v[1](3,1) = 7;
    v[1](4,1) = -2;
    v[1](5,1) = 4;

    v[2](0,0) = 1;
    v[2](1,0) = -1;
    v[2](2,0) = 0;
    v[2](3,0) = 0;
    v[2](4,0) = 3;
    v[2](5,0) = -2;

    v[2](0,1) = 0;
    v[2](1,1) = -1;
    v[2](2,1) = 1;
    v[2](3,1) = 4;
    v[2](4,1) = 2;
    v[2](5,1) = 6;

    Eigen::VectorXd ESS = ess(v_ref);
}

} // namespace math
} // namespace ppl
