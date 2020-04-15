#pragma once
#include <autoppl/expression/rv_tag.hpp>
#include <autoppl/expression/uniform.hpp>
#include <autoppl/expression/model.hpp>

namespace ppl {

// Builds an EqNode to associate tag with dist.
// Ex. x |= uniform(0,1)
template <class DataType, class DistType>
constexpr inline auto operator|=(rv_tag<DataType>& tag,
                                 DistType dist)
{
    return EqNode<rv_tag<DataType>, DistType>(tag, dist);
}

} // namespace ppl
