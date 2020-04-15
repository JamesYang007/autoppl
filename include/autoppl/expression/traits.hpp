#pragma once

namespace ppl {

/*
 * The following classes list member aliases that
 * any such template parameter types should have.
 * Users should rely on these classes to grab member aliases.
 */

template <class TagType>
struct tag_traits
{
    using value_t = typename TagType::value_t;
    using pointer_t = typename TagType::pointer_t;
    using state_t = typename TagType::state_t;
};

template <class DistType>
struct dist_traits
{
    using value_t = typename DistType::value_t;
    using dist_value_t = typename DistType::dist_value_t;
};

template <class NodeType>
struct node_traits
{
    using dist_value_t = typename NodeType::dist_value_t;
};

} // namespace ppl
