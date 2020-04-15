#pragma once
#include <type_traits>
#include <functional>
#include <optional>
#include <autoppl/expression/traits.hpp>

namespace ppl {
namespace details {

template <class Iter>
struct IdentityTagFunctor
{
    using value_t = typename std::iterator_traits<Iter>::value_type;
    value_t& operator()(value_t& tag)
    { return tag; }
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
    using dist_t = DistType;
    using dist_value_t = typename dist_traits<dist_t>::dist_value_t;

    EqNode(const tag_t& tag, 
           const dist_t& dist) noexcept
        : orig_tag_cref_{tag}
        , comp_tag_cref_{}
        , dist_{dist}
    {}

    /*
     * Binds computation-required data with this model.
     * Underlying reference to computation data for this random variable
     * will reference the next data (*begin).
     * The next data (*begin) will be initialized with original tag.
     * The functor getter MUST return an lvalue reference to the tag.
     * TODO: set up compile-time checks for this ^.
     * The tag must be the same type as tag_t.
     */
    template <class Iter, class F = details::IdentityTagFunctor<Iter>>
    Iter bind_comp_data(Iter begin, Iter end, F getter = F()) 
    {
        assert(begin != end);                   // MUST have a value to get
        getter(*begin) = orig_tag_cref_.get();  // initialize comp data
        comp_tag_cref_ = getter(*begin);        // set reference to comp data
        return ++begin;
    }

    /*
     * Compute pdf of underlying distribution with underlying value.
     * Assumes that underlying value has been assigned properly.
     */
    dist_value_t pdf() const
    { return dist_.pdf(comp_tag_cref_->get().get_value()); }

    /*
     * Compute log-pdf of underlying distribution with underlying value.
     * Assumes that underlying value has been assigned properly.
     */
    dist_value_t log_pdf() const
    { return dist_.log_pdf(comp_tag_cref_->get().get_value()); }

private:
    using tag_cref_t = std::reference_wrapper<const tag_t>;
    using opt_tag_cref_t = std::optional<tag_cref_t>;
    
    tag_cref_t orig_tag_cref_;      // (const) reference of the original tag since 
                                    // any configuration may be changed until right before update 
    opt_tag_cref_t comp_tag_cref_;  // reference the tag needed in computation 
    dist_t dist_;                   // distribution associated with tag
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
             const right_node_t& rhs) noexcept
        : left_node_{lhs}
        , right_node_{rhs}
    {}

    /*
     * Binds computational data in order from left to right.
     * In other words, same order as user would list the model expressions.
     */
    template <class Iter, class F = details::IdentityTagFunctor<Iter>>
    Iter bind_comp_data(Iter begin, Iter end, F getter = F()) 
    {
        Iter new_begin = left_node_.bind_comp_data(begin, end, getter);
        return right_node_.bind_comp_data(new_begin, end, getter);
    }

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
