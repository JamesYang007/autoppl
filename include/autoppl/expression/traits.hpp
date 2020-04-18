#pragma once
#include <type_traits>

namespace ppl {

/*
 * The following classes list member aliases that
 * any such template parameter types should have.
 * Users should rely on these classes to grab member aliases.
 */

template <class VarType>
struct var_traits
{
    using value_t = typename VarType::value_t;
    using pointer_t = typename VarType::pointer_t;
    using state_t = typename VarType::state_t;

    // TODO may have to move this to a different class for compile-time checking
    static_assert(std::is_convertible_v<VarType, value_t>);
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
