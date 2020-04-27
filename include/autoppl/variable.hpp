#pragma once
#include <autoppl/util/var_traits.hpp>
#include <autoppl/util/dist_expr_traits.hpp>
#include <vector>
#include <initializer_list>
#include <cassert>

namespace ppl {

/*
 * Param docs
 * 
 */

template <class ValueType>
struct Param : util::ParamLike<Param<ValueType>> {
    using value_t = ValueType;
    using pointer_t = value_t*;
    using const_pointer_t = const value_t*;

    Param(value_t value, pointer_t storage_ptr) noexcept
        : value_{value}, storage_ptr_{storage_ptr} {}

    Param(pointer_t storage_ptr) noexcept
        : Param(0., storage_ptr) {}

    Param(value_t value) noexcept
        : Param(value, nullptr) {}

    Param() noexcept
        : Param(0., nullptr) {}

    void set_value(value_t value) { value_ = value; }

    constexpr size_t size() const { return 1; }
    value_t get_value(size_t) const {
        return value_;
    }

    void set_storage(pointer_t storage_ptr) { storage_ptr_ = storage_ptr; }
    pointer_t get_storage() { return storage_ptr_; }
    const_pointer_t get_storage() const { return storage_ptr_; }

   private:
    value_t value_;  // store value associated with var
    pointer_t storage_ptr_;        // points to beginning of storage
                                   // storage is assumed to be contiguous
};

/* 
 * Data is a light-weight structure that represents a set of samples from an observed random variable.
 * It acts as an intermediate layer of communication between
 * a model expression and the users, who must supply storage of values associated with this var.
 */
template <class ValueType>
struct Data : util::DataLike<Data<ValueType>>
{
    using value_t = ValueType;
    using pointer_t = value_t*;
    using const_pointer_t = const value_t*;

    template <typename iterator>
    Data(iterator begin, iterator end) noexcept
        : values_{begin, end} {}

    Data(std::initializer_list<value_t> values) noexcept
        : Data(values.begin(), values.end()) {}

    Data(value_t value) noexcept
        : values_{{value}} {}

    Data() noexcept : values_{} {}

    size_t size() const { return values_.size(); }

    value_t get_value(size_t i) const { 
        assert((i >= 0) && (i < size()));
        return values_[i]; 
    }

    void observe(value_t value) { values_.push_back(value); }
    void clear() { values_.clear(); }

private:
    std::vector<value_t> values_;             // store value associated with var
};

// Useful aliases
using cont_var = Data<util::cont_param_t>; // continuous RV var
using disc_var = Data<util::disc_param_t>; // discrete RV var

} // namespace ppl
