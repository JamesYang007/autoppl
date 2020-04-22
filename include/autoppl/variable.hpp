#pragma once
#include <autoppl/util/var_traits.hpp>
#include <autoppl/util/dist_expr_traits.hpp>

namespace ppl {

/*
 * The possible states for a var.
 * By default, all vars should be considered as a parameter.
 * TODO: maybe move in a different file?
 */
enum class var_state : bool {
    data,
    parameter
};

/* 
 * Variable is a light-weight structure that represents a univariate random variable.
 * It acts as an intermediate layer of communication between
 * a model expression and the users, who must supply storage of values associated with this var.
 */
template <class ValueType>
struct Variable
{
    using value_t = ValueType;
    using pointer_t = value_t*;
    using const_pointer_t = const value_t*;
    using state_t = var_state;
	using binop_result_t = ValueType;

    // constructors
    Variable(value_t value, 
             pointer_t storage_ptr,
             state_t state) noexcept
        : value_{value}
        , storage_ptr_{storage_ptr}
        , state_{state}
    {}

    Variable(pointer_t storage_ptr) noexcept
        : Variable(0, storage_ptr, state_t::parameter)
    {}

    Variable(value_t value) noexcept
        : Variable(value, nullptr, state_t::data) {}

    Variable() noexcept
        : Variable(0, nullptr, state_t::parameter)
    {}

    void set_value(value_t value) { value_ = value; }
    value_t get_value() const { return value_; }

    void set_storage(pointer_t storage_ptr) { storage_ptr_ = storage_ptr; }
    pointer_t get_storage() { return storage_ptr_; }
    const_pointer_t get_storage() const { return storage_ptr_; }

    void set_state(state_t state) { state_ = state; }
    state_t get_state() const { return state_; }

    explicit operator value_t () const { return get_value(); }

    /*
     * Sets underlying value to "value".
     * Additionally modifies the var to be considered as data.
     * Equivalent to calling set_value(value) then set_state(state).
     */
    void observe(value_t value)
    {
        set_value(value);
        set_state(state_t::data);
    }

private:
    value_t value_;             // store value associated with var
    pointer_t storage_ptr_;     // points to beginning of storage 
                                // storage is assumed to be contiguous
    state_t state_;             // state to determine if data or param
};

// Useful aliases
using cont_var = Variable<util::cont_param_t>; // continuous RV var
using disc_var = Variable<util::disc_param_t>; // discrete RV var

} // namespace ppl
