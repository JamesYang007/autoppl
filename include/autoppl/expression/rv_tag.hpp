#pragma once

namespace ppl {

/*
 * The possible states for a tag.
 * By default, all tags should be considered as a parameter.
 * TODO: maybe move in a different file?
 */
enum class tag_state : bool {
    data,
    parameter
};

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
    using const_pointer_t = const value_t*;
    using state_t = tag_state;

    // constructors
    rv_tag(value_t value, 
           pointer_t storage_ptr) noexcept
        : value_{value}
        , storage_ptr_{storage_ptr}
        , state_{state_t::parameter}
    {}

    rv_tag(pointer_t storage_ptr) noexcept
        : rv_tag(0, storage_ptr)
    {}

    rv_tag() noexcept
        : rv_tag(0, nullptr)
    {}

    void set_value(value_t value) { value_ = value; }
    value_t get_value() const { return value_; }

    void set_storage(pointer_t storage_ptr) { storage_ptr_ = storage_ptr; }
    pointer_t get_storage() { return storage_ptr_; }
    const_pointer_t get_storage() const { return storage_ptr_; }

    void set_state(state_t state) { state_ = state; }
    state_t get_state() const { return state_; }

    /*
     * Sets underlying value to "value".
     * Additionally modifies the tag to be considered as data.
     * Equivalent to calling set_value(value) then set_state(state).
     */
    void observe(value_t value)
    {
        set_value(value);
        set_state(state_t::data);
    }

private:
    value_t value_;             // store value associated with tag
    pointer_t storage_ptr_;     // points to beginning of storage 
                                // storage is assumed to be contiguous
    state_t state_;             // state to determine if data or param
};

// Useful aliases
using cont_rv_tag = rv_tag<double>; // continuous RV tag
using disc_rv_tag = rv_tag<int>;    // discrete RV tag

} // namespace ppl
