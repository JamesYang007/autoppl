#pragma once
#include <type_traits>
#include <functional>
#include <autoppl/expression/traits.hpp>

namespace ppl {

/*
 * This class represents a "node" in the model expression
 * that relates a tag with a distribution.
 */
template <class TagType, class DistType>
struct EqNode
{
    using tag_t = TagType;
    using dist_t = DistType;
    using dist_value_t = typename dist_traits<dist_t>::dist_value_t;

    EqNode(const tag_t& tag, 
           const dist_t& dist) noexcept
        : tag_{}
        , tag_cref_{tag}
        , dist_{dist}
    {}

    /*
     * Updates underlying tag by copying from referenced tag.
     * This must be called before the model is used, if
     * any members of the referenced tag changed.
     */
    void update()
    { tag_ = tag_cref_; }

    /*
     * Compute pdf of underlying distribution with underlying value.
     * Assumes that underlying value has been assigned properly.
     */
    dist_value_t pdf() const
    { return dist_.pdf(tag_.get_value()); }

    /*
     * Compute log-pdf of dist_ with value x.
     * Assumes that value_ has been assigned with a proper value.
     */
    dist_value_t log_pdf() const
    { return dist_.log_pdf(tag_.get_value()); }

private:
    using tag_cref_t = std::reference_wrapper<const tag_t>;
    
    tag_t tag_;             // cache optimization
    tag_cref_t tag_cref_;   // (const) reference of the tag since any configuration
                            // may be changed until right before update 
    dist_t dist_;           // distribution associated with tag
};

/*
 * This class represents a "node" in a model expression that
 * "glues" two sub-model expressions.
 */
template <class LHSNodeType, class RHSNodeType>
struct GlueNode
{
    using left_node_t = LHSNodeType;
    using right_node_t = RHSNodeType;
    using dist_value_t = std::common_type_t<
        typename node_traits<left_node_t>::dist_value_t,
        typename node_traits<right_node_t>::dist_value_t
            >;

    GlueNode(const left_node_t& lhs,
             const right_node_t& rhs)
        : left_node_{lhs}
        , right_node_{rhs}
    {}

    /*
     * Updates left node first then right node by calling "update" on each.
     * This must be called before the model is used, 
     * if either left or right must be updated.
     */
    void update()
    { left_node_.update(); right_node_.update(); }

    /*
     * Computes left node joint pdf then right node joint pdf
     * and returns the product of the two.
     */
    dist_value_t pdf() const
    { return left_node_.pdf() * right_node_.pdf(); }

    /*
     * Computes left node joint log-pdf then right node joint log-pdf
     * and returns the sum of the two.
     */
    dist_value_t log_pdf() const
    { return left_node_.log_pdf() + right_node_.log_pdf(); }

private:
    left_node_t left_node_;
    right_node_t right_node_;
};

/////////////////////////////////////////////////////////
// Operator overloads
/////////////////////////////////////////////////////////

// TODO: all these template parameters should be constrained 
// with concepts!

/*
 * Builds an EqNode to associate tag with dist.
 * Ex. x |= uniform(0,1)
 */
template <class TagType, class DistType>
constexpr inline auto operator|=(const TagType& tag,
                                 const DistType& dist)
{
    return EqNode<TagType, DistType>(tag, dist);
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
