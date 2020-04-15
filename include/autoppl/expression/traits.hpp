#pragma once

namespace ppl {

/*
 * This class lists all member aliases that
 * any tag types should have, in particular, rv_tag.
 * Users should rely on this class to grab member aliases.
 */
template <class TagType>
struct tag_traits
{
    using value_t = typename TagType::value_t;
    using pointer_t = typename TagType::pointer_t;
};

/*
 * This class lists all member aliases that
 * any distribution types should have.
 * Users should rely on this class to grab member aliases.
 */
template <class DistType>
struct dist_traits
{
    using value_t = typename DistType::value_t;
    using dist_value_t = typename DistType::dist_value_t;
};

} // namespace ppl
