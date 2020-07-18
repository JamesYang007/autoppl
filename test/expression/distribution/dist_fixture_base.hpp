#pragma once
#include <array>
#include <autoppl/expression/variable/data.hpp>
#include <autoppl/expression/variable/param.hpp>

namespace ppl {
namespace expr {

template <class ValueType>
struct dist_fixture_base  {
protected:
    static constexpr size_t vec_size = 3;
    static constexpr size_t offset_max_size = 3;

    using value_t = ValueType;
    using pointer_t = value_t*;
    using vec_t = std::vector<value_t>;
    using vec_pointer_t = std::array<pointer_t, 3*vec_size>;

    using dv_scl_t = DataView<value_t, ppl::scl>;
    using dv_vec_t = DataView<vec_t, ppl::vec>;
    using pv_scl_t = ParamView<pointer_t, ppl::scl>;
    using pv_vec_t = ParamView<vec_pointer_t, ppl::vec>;
    using id_t = typename util::var_traits<dv_scl_t>::id_t;
    using index_t = typename util::param_traits<pv_scl_t>::index_t;
    using ad_vec_t = std::vector<ad::Var<value_t>>;

    std::array<index_t, offset_max_size> offsets = {0};
    vec_pointer_t storage = {nullptr}; 
    std::vector<ad::Var<value_t>> cache;
};

} // namespace expr
} // namespace ppl
