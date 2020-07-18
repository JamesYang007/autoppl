#include "gtest/gtest.h"
#include <autoppl/expression/variable/data.hpp>
#include <autoppl/util/traits/var_traits.hpp>
#include <autoppl/util/iterator/counting_iterator.hpp>

namespace ppl {
namespace expr {

struct data_fixture : ::testing::Test {
protected:
    using value_type = double;
    using vec_type = std::vector<value_type>;
    using mat_type = arma::Mat<value_type>;
    using dview_scl_t = DataView<value_type, ppl::scl>;
    using dview_vec_t = DataView<vec_type, ppl::vec>;
    using dview_mat_t = DataView<mat_type, ppl::mat>;
    using d_scl_t = Data<value_type, ppl::scl>;
    using d_vec_t = Data<value_type, ppl::vec>;

    static constexpr value_type defval1 = 1.0;
    static constexpr value_type defval2 = 2.0;
    static constexpr size_t size1 = 7;
    static constexpr size_t size2 = 17;

    value_type d1 = defval1;
    value_type d2 = defval2;

    vec_type values1;
    vec_type values2;

    data_fixture()
        : values1(size1)
        , values2(size2)
    {
        std::transform(util::counting_iterator<>(0),
                       util::counting_iterator<>(size1),
                       values1.begin(),
                       [=](auto i) { return i + defval1; });

        std::transform(util::counting_iterator<>(0),
                       util::counting_iterator<>(size2),
                       values2.begin(),
                       [=](auto i) { return i + defval2; });
    }
};

TEST_F(data_fixture, type_check)
{
    static_assert(util::is_data_v<dview_scl_t>);
    static_assert(util::is_scl_v<dview_scl_t>);

    static_assert(util::is_data_v<dview_vec_t>);
    static_assert(util::is_vec_v<dview_vec_t>);

    static_assert(util::is_data_v<dview_mat_t>);
    static_assert(util::is_mat_v<dview_mat_t>);

    static_assert(util::is_data_v<d_scl_t>);
    static_assert(util::is_scl_v<d_scl_t>);

    static_assert(util::is_data_v<d_vec_t>);
    static_assert(util::is_vec_v<d_vec_t>);
}

////////////////////////////////////////
// DataView: scl
////////////////////////////////////////

TEST_F(data_fixture, dview_scl_value)
{
    dview_scl_t view(d1);

    // all parameters should not matter
    // this is was just to match API for variable expressions
    // data already views its own values
    EXPECT_DOUBLE_EQ(view.value(values1, 0), d1);
    EXPECT_DOUBLE_EQ(view.value(values1, 1), d1);
    EXPECT_DOUBLE_EQ(view.value(values2, 2), d1);
}

TEST_F(data_fixture, dview_scl_size)
{
    dview_scl_t view(d1);
    EXPECT_EQ(view.size(), 1ul);
}

TEST_F(data_fixture, dview_scl_to_ad)
{
    dview_scl_t view(d1);
    // both parameters are ignored
    auto expr = view.to_ad(0,0); 
    EXPECT_DOUBLE_EQ(ad::evaluate(expr), defval1);
}

////////////////////////////////////////
// DataView: vec
////////////////////////////////////////

TEST_F(data_fixture, dview_vec_value)
{
    dview_vec_t view(values1);
    // passed in values should not matter at all
    // data already views its own values
    // the index matters though
    EXPECT_DOUBLE_EQ(view.value(values2, 0), values1[0]);
    EXPECT_DOUBLE_EQ(view.value(values2, 1), values1[1]);
    EXPECT_DOUBLE_EQ(view.value(values2, 2), values1[2]);
}

TEST_F(data_fixture, dview_vec_size)
{
    dview_vec_t view(values1);
    EXPECT_EQ(view.size(), values1.size());
}

TEST_F(data_fixture, dview_vec_to_ad)
{
    dview_vec_t view(values1);
    // only the last argument is not ignored
    auto expr = view.to_ad(0,0,3); 
    EXPECT_DOUBLE_EQ(ad::evaluate(expr), values1[3]);
}

}  // namespace expr
}  // namespace ppl
