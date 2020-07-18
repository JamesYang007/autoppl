#pragma once
#include <cassert>
#include <vector>
#include <autoppl/util/traits/var_traits.hpp>
#include <autoppl/util/traits/shape_traits.hpp>
#include <autoppl/util/functional.hpp>

#define PPL_PARAMVIEW_SHAPE_UNSUPPORTED \
    "Unsupported shape for ParamView. "
#define PPL_PARAM_SHAPE_UNSUPPORTED \
    "Unsupported shape for Param. "

namespace ppl {

/**
 * ParamView is a class that views the storage pointer(s).
 * Note that it is viewing a storage pointer (or vector of pointers) and not the storage itself.
 * Users will likely not need to create these objects directly.
 * The easier-to-use Param class template will be used.
 * When constructing a model expression, both types will be converted to a ParamView.
 *
 * Specializations when ShapeType is not one of (ppl::scl, ppl::vec, or ppl::mat)
 * is disabled.
 *
 * @tparam PointerType  pointer type for storage pointer to view when ShapeType
 *                      is ppl::scl. It is a vector of pointer type when ShapeType
 *                      is ppl::vec.
 * @tparam ShapeType    shape of the object it is viewing. 
 *                      Currently does not support ppl::mat.
 */

template <class PointerType
        , class ShapeType = ppl::scl>
struct ParamView
{
    static_assert(util::is_scl_v<ShapeType> ||
                  util::is_vec_v<ShapeType>,
                  PPL_PARAMVIEW_SHAPE_UNSUPPORTED);
};

template <class PointerType>
struct ParamView<PointerType, ppl::scl>:
    util::VarExprBase<ParamView<PointerType, ppl::scl>>,
    util::ParamBase<ParamView<PointerType, ppl::scl>> 
{
    using pointer_t = PointerType;
    using value_t = std::remove_const_t<
        std::remove_pointer_t<pointer_t> >;
    using const_pointer_t = const value_t*;
    using const_storage_pointer_t = const pointer_t*;
    using id_t = const void*;
    using shape_t = ppl::scl;
    using index_t = uint32_t;
    static constexpr bool has_param = true;
    static constexpr size_t fixed_size = 1;

    // Note: id may need to be provided when subscripting
    ParamView(index_t& offset, 
              const pointer_t& storage_ptr,
              id_t id,
              index_t rel_offset = 0) noexcept
        : offset_ptr_{&offset} 
        , rel_offset_{rel_offset}
        , storage_ptr_ptr_{&storage_ptr} 
        , id_{id}
    {}

    ParamView(index_t& offset, 
              const pointer_t& storage_ptr,
              index_t rel_offset = 0) noexcept
        : ParamView(offset, storage_ptr, this, rel_offset)
    {}

    template <class VecType
            , class F = util::identity>
    auto& value(VecType& vars,
                size_t=0,
                F f = F()) const 
    { 
        return f.template operator()<value_t>(
            vars[*offset_ptr_ + rel_offset_]); 
    }

    template <class VecType
            , class F = util::identity>
    auto value(const VecType& vars,
               size_t=0,
               F f = F()) const 
    { 
        return f.template operator()<value_t>(
            vars[*offset_ptr_ + rel_offset_]); 
    }
    
    constexpr size_t size() const { return fixed_size; }

    pointer_t storage(size_t=0) const 
    { return *storage_ptr_ptr_; }

    id_t id() const { return id_; }

    template <class VecType>
    auto to_ad(const VecType& vars,
               const VecType&,
               size_t=0) const 
    { return vars[*offset_ptr_ + rel_offset_]; }

    index_t set_offset(index_t offset) { 
        *offset_ptr_ = offset; 
        return offset + this->size();
    }

    index_t set_cache_offset(index_t idx) const 
    { return idx; }

private:
    index_t* const offset_ptr_;
    const index_t rel_offset_;
    const_storage_pointer_t storage_ptr_ptr_;
    const id_t id_; 
};

template <class VecType>
struct ParamView<VecType, ppl::vec>:
    util::VarExprBase<ParamView<VecType, ppl::vec>>,
    util::ParamBase<ParamView<VecType, ppl::vec>> 
{
    using vec_t = VecType;
    using pointer_t = typename VecType::value_type;
    using value_t = std::remove_const_t<
        std::remove_pointer_t<pointer_t> >;
    using const_pointer_t = const value_t*;
    using shape_t = ppl::vec;
    using index_t = uint32_t;
    using id_t = const void*;
    static constexpr bool has_param = true;
    static constexpr size_t fixed_size = 0;

    ParamView(index_t& offset,
              const vec_t& storages,
              index_t size) noexcept
        : offset_ptr_{&offset} 
        , storages_ptr_{&storages} 
        , id_{this}
        , size_{size}
    {}

    template <class PVecType
            , class F = util::identity>
    auto& value(PVecType& vars,
                size_t i,
                F f = F()) const 
    { 
        return f.template operator()<value_t>(
                vars[*offset_ptr_ + i]); 
    }

    template <class PVecType
            , class F = util::identity>
    auto value(const PVecType& vars,
               size_t i,
               F f = F()) const 
    { 
        return f.template operator()<value_t>(
                vars[*offset_ptr_ + i]); 
    }

    size_t size() const { return size_; }

    pointer_t storage(size_t i) const 
    { return (*storages_ptr_)[i]; }

    id_t id() const { return id_; }

    template <class VecADVarType>
    auto to_ad(const VecADVarType& vars,
               const VecADVarType&,
               size_t i) const 
    { return vars[*offset_ptr_ + i]; }

    index_t set_offset(index_t offset) { 
        *offset_ptr_ = offset; 
        return offset + this->size();
    }

    index_t set_cache_offset(index_t idx) const 
    { return idx; }
    
    auto operator[](index_t i) { 
        return ParamView<pointer_t, ppl::scl>(
                *offset_ptr_, 
                (*storages_ptr_)[i], 
                id_,
                i);
    }

private:
    index_t* const offset_ptr_; // note: underlying offset CAN be changed by viewer
    const vec_t* storages_ptr_;
    const id_t id_;
    const index_t size_;
};

/**
 * Param is a class template wrapping a ParamView for user-friendly usage.
 * It owns a container of storage pointers which the user specifies
 * to point to where samples should go.
 * A Param is a ParamView (it views itself).
 * Similar to ParamView, it must be given a shape tag.
 *
 * @tparam ValueType    underlying value type (usually double or int)
 * @tparam ShapeType    one of the three shape tags.
 *                      Currently, ppl::mat is not supported.
 */

template <class ValueType
        , class ShapeType = ppl::scl>
struct Param
{
    static_assert(util::is_scl_v<ShapeType> ||
                  util::is_vec_v<ShapeType>,
                  PPL_PARAM_SHAPE_UNSUPPORTED);
};

template <class ValueType>
struct Param<ValueType, ppl::scl>: 
    ParamView<ValueType*, ppl::scl>,
    util::VarExprBase<Param<ValueType, ppl::scl>>,
    util::ParamBase<Param<ValueType, ppl::scl>>
{
    using base_t = ParamView<ValueType*, ppl::scl>;
    using typename base_t::value_t;
    using typename base_t::pointer_t;
    using typename base_t::const_pointer_t;
    using typename base_t::id_t;
    using typename base_t::index_t;
    using typename base_t::shape_t;
    using base_t::value;
    using base_t::size;
    using base_t::storage;
    using base_t::to_ad;
    using base_t::id;
    using base_t::set_offset;

    Param(pointer_t ptr=nullptr) noexcept
        : base_t(offset_, storage_ptr_)
        , offset_(0) 
        , storage_ptr_(ptr)
    {}

    pointer_t& storage(size_t=0) { return storage_ptr_; }

private:
    index_t offset_;
    pointer_t storage_ptr_;
};

template <class ValueType>
struct Param<ValueType, ppl::vec> : 
    ParamView<std::vector<ValueType*>, ppl::vec>,
    util::VarExprBase<Param<ValueType, ppl::vec>>,
    util::ParamBase<Param<ValueType, ppl::vec>>
{
    using base_t = ParamView<std::vector<ValueType*>, ppl::vec>;
    using typename base_t::value_t;
    using typename base_t::pointer_t;
    using typename base_t::const_pointer_t;
    using typename base_t::id_t;
    using typename base_t::index_t;
    using typename base_t::shape_t;
    using base_t::value;
    using base_t::size;
    using base_t::storage;
    using base_t::to_ad;
    using base_t::id;
    using base_t::set_offset;

    Param(size_t n)
        : base_t(offset_, storage_ptrs_, n)
        , storage_ptrs_(n, nullptr)
    {}

    Param(std::initializer_list<pointer_t> ptrs) noexcept
        : base_t(offset_, storage_ptrs_, ptrs.size())
        , offset_(0)
        , storage_ptrs_(ptrs)
    {}

    pointer_t& storage(size_t i) { return storage_ptrs_[i]; }

private:
    index_t offset_; 
    std::vector<pointer_t> storage_ptrs_;
};

// TODO: Specialization: mat-like
// TODO: ParamFixed

} // namespace ppl

#undef PPL_PARAMVIEW_SHAPE_UNSUPPORTED
#undef PPL_PARAM_SHAPE_UNSUPPORTED
