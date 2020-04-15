#pragma once
#include <autoppl/expression/rv_tag.hpp>
#include <autoppl/expression/uniform.hpp>
#include <autoppl/expression/model.hpp>

namespace ppl {

// TODO: all these template parameters should be constrained 
// with concepts!

/*
 * Builds an EqNode to associate tag with dist.
 * Ex. x |= uniform(0,1)
 */
template <class DataType, class DistType>
constexpr inline auto operator|=(const rv_tag<DataType>& tag,
                                 const DistType& dist)
{
    return EqNode<rv_tag<DataType>, DistType>(tag, dist);
}

/*
 * Builds a GlueNode to "glue" the left expression with the right.
 * Ex. (x |= uniform(0,1), y |= uniform(0, 2))
 */
template <class LHSNodeType, class RHSNodeType>
constexpr inline auto operator,(const LHSNodeType& lhs,
                                const RHSNodeType& rhs)
{
    return GlueNode<LHSNodeType, RHSNodeType>(lhs, rhs);
}

} // namespace ppl
