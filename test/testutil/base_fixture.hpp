#pragma once
#include <autoppl/util/packs/ptr_pack.hpp>
#include <autoppl/expression/variable/data.hpp>
#include <autoppl/expression/variable/param.hpp>
#include <autoppl/expression/variable/tparam.hpp>
#include <autoppl/expression/variable/constant.hpp>

namespace ppl {

template <class T>
struct base_fixture
{
protected:
    using value_t = T;

    using scl_tp_t = TParam<value_t, ppl::scl>;
    using vec_tp_t = TParam<value_t, ppl::vec>;
    using mat_tp_t = TParam<value_t, ppl::mat>;
    using scl_tpv_t = TParamView<value_t, ppl::scl>;
    using vec_tpv_t = TParamView<value_t, ppl::vec>;
    using mat_tpv_t = TParamView<value_t, ppl::mat>;

    using scl_p_t = Param<value_t, ppl::scl>;
    using vec_p_t = Param<value_t, ppl::vec>;
    using mat_p_t = Param<value_t, ppl::mat>;
    using scl_pv_t = ParamView<value_t, ppl::scl>;
    using vec_pv_t = ParamView<value_t, ppl::vec>;
    using mat_pv_t = ParamView<value_t, ppl::mat>;

    using scl_d_t = Data<value_t, ppl::scl>;
    using vec_d_t = Data<value_t, ppl::vec>;
    using mat_d_t = Data<value_t, ppl::mat>;
    using scl_dv_t = DataView<value_t, ppl::scl>;
    using vec_dv_t = DataView<value_t, ppl::vec>;
    using mat_dv_t = DataView<value_t, ppl::mat>;

    using scl_c_t = expr::var::Constant<value_t, ppl::scl>;
    using vec_c_t = expr::var::Constant<value_t, ppl::vec>;
    using mat_c_t = expr::var::Constant<value_t, ppl::mat>;

    using id_t = typename util::var_traits<scl_dv_t>::id_t;
    using info_pack_t = details::ParamInfoPack;
    using info_tpack_t = details::TParamInfoPack;
    using offset_pack_t = util::OffsetPack;
    using ptr_pack_t = util::PtrPack<
        value_t*, value_t*, value_t*, value_t*, value_t*>;

    std::vector<value_t> val_buf;

    ptr_pack_t ptr_pack;

    template <class ExprType>
    void bind(ExprType& expr) 
    {
        ptr_pack_t pack(val_buf.data());
        expr.bind(pack);
    }
};

} // namespace ppl
