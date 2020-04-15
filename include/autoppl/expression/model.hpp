#pragma once
#include <functional>
#include <autoppl/expression/traits.hpp>

namespace ppl {
namespace details {

/*
 * The possible states for a node expression.
 * By default, all nodes should be a parameter.
 */
enum class node_state : bool {
    data,
    parameter
};

} // namespace details

/*
 * This class represents a "node" in the model expression
 * that relates a tag with a distribution.
 */
template <class TagType, class DistType>
struct EqNode
{
    using tag_t = TagType;
    using tag_cref_t = std::reference_wrapper<const tag_t>;
    using value_t = typename tag_traits<tag_t>::value_t;
    using pointer_t = typename tag_traits<tag_t>::pointer_t;

    using state_t = details::node_state;

    using dist_t = DistType;
    using gen_value_t = typename tag_traits<tag_t>::value_t;
    using dist_value_t = typename dist_traits<dist_t>::dist_value_t;

    EqNode(const tag_t& tag, dist_t dist) noexcept
        : value_{}
        , tag_{}
        , state_{state_t::parameter}
        , tag_cref_{tag}
        , dist_{dist}
    {}

    /*
     * Compute pdf of dist_ with value x.
     */
    dist_value_t pdf(gen_value_t x) const
    { return dist_.pdf(x); }

    /*
     * Compute log-pdf of dist_ with value x.
     */
    dist_value_t log_pdf(gen_value_t x) const
    { return dist_.log_pdf(x); }

private:
    // cache optimization
    value_t value_;         // value to store during computation
    tag_t tag_;             // tag to manage the storage: get/put values

    state_t state_;         // state to determine if data or param
    tag_cref_t tag_cref_;   // (const) reference of the tag since
                            // storage of tag may be bound right before some computation
    dist_t dist_;           // distribution associated with tag
};

} // namespace ppl
