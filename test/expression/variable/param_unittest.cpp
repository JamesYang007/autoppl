#include "gtest/gtest.h"
#include <unordered_map>
#include <array>
#include <autoppl/expression/variable/param.hpp>
#include <autoppl/util/traits/var_traits.hpp>
#include <autoppl/util/iterator/counting_iterator.hpp>

namespace ppl {
namespace expr {

struct param_fixture : ::testing::Test {
protected:
    using value_type = double;
    using pointer_t = value_type*;
    using vec_pointer_t = std::vector<pointer_t>;

    using pview_scl_t = ParamView<pointer_t, ppl::scl>;
    using pview_vec_t = ParamView<vec_pointer_t, ppl::vec>;
    using p_scl_t = Param<value_type, ppl::scl>;
    using p_vec_t = Param<value_type, ppl::vec>;

    using index_t = typename util::param_traits<pview_scl_t>::index_t;

    static constexpr value_type defval1 = 1.0;
    static constexpr value_type defval2 = 2.0;
    static constexpr size_t size1 = 7;
    static constexpr size_t size2 = 17;

    // hypothetical storage: one sample for each param value
    std::array<value_type, size1> storage1 = {0};
    std::array<value_type, size2> storage2 = {0};

    // hypothetical parameter values
    std::vector<value_type> values1;    
    std::vector<value_type> values2;

    // hypothetical storage ptrs for sample
    vec_pointer_t storage_ptrs1;    
    vec_pointer_t storage_ptrs2;

    // hypothetical offsets
    index_t offset = 0;

    param_fixture() 
        : values1(size1)
        , values2(size2)
        , storage_ptrs1(size1)
        , storage_ptrs2(size2)
    {
        std::transform(util::counting_iterator<>(0),
                       util::counting_iterator<>(size1),
                       values1.begin(),
                       [=](auto i) { return i + defval1; });

        std::transform(util::counting_iterator<>(0),
                       util::counting_iterator<>(size2),
                       values2.begin(),
                       [=](auto i) { return i + defval2; });

        std::transform(storage1.begin(),
                       storage1.end(),
                       storage_ptrs1.begin(),
                       [](auto& x) { return &x; });

        std::transform(storage2.begin(),
                       storage2.end(),
                       storage_ptrs2.begin(),
                       [](auto& x) { return &x; });
    }
};

TEST_F(param_fixture, type_check)
{
    static_assert(util::is_param_v<pview_scl_t>);
    static_assert(util::is_param_v<pview_vec_t>);
    static_assert(util::is_param_v<p_scl_t>);
    static_assert(util::is_param_v<p_vec_t>);
}

////////////////////////////////////////
// DataView: scl
////////////////////////////////////////

TEST_F(param_fixture, pview_scl_value)
{
    auto&& s1 = storage_ptrs1[0];
    pview_scl_t view(offset, s1, 1);

    // last parameter should not matter
    EXPECT_DOUBLE_EQ(view.value(values1, 0), values1[1]);
    EXPECT_DOUBLE_EQ(view.value(values1, 1), values1[1]);
    EXPECT_DOUBLE_EQ(view.value(values1, 2), values1[1]);

    // able to view a different array of values
    EXPECT_DOUBLE_EQ(view.value(values2, 0), values2[1]);
    EXPECT_DOUBLE_EQ(view.value(values2, 1), values2[1]);
    EXPECT_DOUBLE_EQ(view.value(values2, 2), values2[1]);
}

TEST_F(param_fixture, pview_scl_storage)
{
    auto&& s1 = storage_ptrs1[0];

    pview_scl_t view(offset, s1, 2);
    // parameter should not matter 
    EXPECT_EQ(view.storage(0), s1);
    EXPECT_EQ(view.storage(1), s1);
    EXPECT_EQ(view.storage(2), s1);

    // relative offset should not affect storage
    pview_scl_t view2(offset, s1, 13124);
    EXPECT_EQ(view2.storage(0), s1);
    EXPECT_EQ(view2.storage(1), s1);
    EXPECT_EQ(view2.storage(2), s1);
}

TEST_F(param_fixture, pview_scl_size)
{
    pview_scl_t view(offset, storage_ptrs1[0]);
    EXPECT_EQ(view.size(), 1ul);
}

TEST_F(param_fixture, pview_scl_to_ad)
{
    auto&& s1 = storage_ptrs1[0];
    pview_scl_t view(offset, s1);

    // simply tests if gets correct elt from passed in array
    // last two parameter should be ignored
    const auto& elt = view.to_ad(storage_ptrs1, storage_ptrs1, 0); 
    EXPECT_EQ(elt, s1);

    const auto& elt2 = view.to_ad(storage_ptrs1, storage_ptrs1, 1); 
    EXPECT_EQ(elt2, s1);
}

////////////////////////////////////////
// DataView: vec
////////////////////////////////////////

TEST_F(param_fixture, pview_vec_value)
{
    pview_vec_t view(offset, storage_ptrs1, storage_ptrs1.size());
    // parameter SHOULD matter
    EXPECT_DOUBLE_EQ(view.value(values1, 0), values1[0]);
    EXPECT_DOUBLE_EQ(view.value(values1, 1), values1[1]);
    EXPECT_DOUBLE_EQ(view.value(values1, 2), values1[2]);
}

TEST_F(param_fixture, pview_vec_size)
{
    pview_vec_t view(offset, storage_ptrs1, storage_ptrs1.size());
    EXPECT_EQ(view.size(), size1);
}

TEST_F(param_fixture, pview_vec_storage)
{
    pview_vec_t view(offset, storage_ptrs1, storage_ptrs1.size());
    EXPECT_EQ(view.storage(0), storage_ptrs1[0]);
    EXPECT_EQ(view.storage(1), storage_ptrs1[1]);
    EXPECT_EQ(view.storage(2), storage_ptrs1[2]);
}

TEST_F(param_fixture, pview_vec_to_ad)
{
    pview_vec_t view(offset, storage_ptrs1, storage_ptrs1.size());

    auto elt = view.to_ad(storage_ptrs1, storage_ptrs1, 0);
    EXPECT_EQ(elt, &storage1[0]);

    elt = view.to_ad(storage_ptrs1, storage_ptrs1, 3);
    EXPECT_EQ(elt, &storage1[3]);
}

}  // namespace expr
}  // namespace ppl
