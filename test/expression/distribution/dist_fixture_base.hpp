#pragma once
#include <array>
#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/core/var.hpp>

namespace ppl {
namespace expr {

template <class ValueType>
struct dist_fixture_base: base_fixture<ValueType>
{
protected:
    static constexpr size_t vec_size = 3;
    static constexpr size_t info_max_size = 3;

    using base_t = base_fixture<ValueType>;
    using typename base_t::value_t;
    using typename base_t::info_pack_t;

    using vec_t = std::array<value_t, vec_size>; 
    using mat_t = std::vector<value_t>;
    using ad_vec_t = std::vector<ad::Var<value_t>>;

    std::array<info_pack_t, info_max_size> infos;
};

} // namespace expr
} // namespace ppl
