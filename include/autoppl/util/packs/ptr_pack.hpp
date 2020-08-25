#pragma once
#include <cstddef>
#include <autoppl/util/traits/dist_expr_traits.hpp>

namespace ppl {
namespace util {

template <class UCValPtrType
        , class UCAdjPtrType
        , class TPValPtrType
        , class TPAdjPtrType
        , class CValPtrType>
struct PtrPack
{
    using uc_val_ptr_t = UCValPtrType;
    using c_val_ptr_t = CValPtrType;
    using tp_val_ptr_t = TPValPtrType;

    PtrPack(UCValPtrType _uc_val = nullptr,
              UCAdjPtrType _uc_adj = nullptr,
              TPValPtrType _tp_val = nullptr,
              TPAdjPtrType _tp_adj = nullptr,
              CValPtrType _c_val = nullptr,
              size_t* _v_val = nullptr)
        : uc_val{_uc_val}
        , uc_adj{_uc_adj}
        , tp_val{_tp_val}
        , tp_adj{_tp_adj}
        , c_val{_c_val}
        , v_val{_v_val}
    {}

    UCValPtrType uc_val;
    UCAdjPtrType uc_adj;
    TPValPtrType tp_val;
    TPAdjPtrType tp_adj;
    CValPtrType c_val;
    size_t* v_val;
};

template <class UCValPtrType = std::nullptr_t
        , class UCAdjPtrType = std::nullptr_t 
        , class TPValPtrType = std::nullptr_t
        , class TPAdjPtrType = std::nullptr_t
        , class CValPtrType  = std::nullptr_t>
constexpr inline auto
make_ptr_pack(UCValPtrType _uc_val = nullptr,
              UCAdjPtrType _uc_adj = nullptr,
              TPValPtrType _tp_val = nullptr,
              TPAdjPtrType _tp_adj = nullptr,
              CValPtrType _c_val = nullptr,
              size_t* _v_val = nullptr)
{ 
    return PtrPack<UCValPtrType,
                   UCAdjPtrType,
                   TPValPtrType,
                   TPAdjPtrType,
                   CValPtrType>(_uc_val, _uc_adj, 
                                _tp_val, _tp_adj, 
                                _c_val, _v_val);
}

using cont_ptr_pack_t = 
    PtrPack<util::cont_param_t*, util::cont_param_t*,
            util::cont_param_t*, util::cont_param_t*,
            util::cont_param_t*>;

// Set adjoint pointer type to void* to detect dereferencing at compile-time.
using disc_ptr_pack_t = 
    PtrPack<util::disc_param_t*, void*,
            util::disc_param_t*, void*,
            util::disc_param_t*>;

} // namespace util
} // namespace ppl
