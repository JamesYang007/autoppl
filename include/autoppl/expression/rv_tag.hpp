#pragma once

namespace ppl {

/* 
 * rv_tag is a light-weight structure that represents a univariate random variable.
 * It acts as an intermediate layer of communication between
 * a model expression and the users, who must supply storage of values associated with this tag.
 */
template <class ValueType>
struct rv_tag
{
    using value_t = ValueType;
    using pointer_t = value_t*;

    rv_tag(pointer_t storage_ptr = nullptr) noexcept
        : storage_ptr_{storage_ptr}
    {}

    /*
     * Binds storage pointer to storage_ptr.
     */
    void bind_storage(pointer_t storage_ptr) 
    { storage_ptr_ = storage_ptr; }

private:
    pointer_t storage_ptr_;     // points to beginning of storage
                                // storage is assumed to be contiguous
};

// Useful aliases
using cont_rv_tag = rv_tag<double>; // continuous RV tag
using disc_rv_tag = rv_tag<int>;    // discrete RV tag

} // namespace ppl
