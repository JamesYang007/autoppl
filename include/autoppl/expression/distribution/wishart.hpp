#pragma once
#include <fastad_bits/reverse/stat/wishart.hpp>
#include <autoppl/util/traits/traits.hpp>
#include <autoppl/expression/distribution/dist_utils.hpp>
#include <autoppl/math/density.hpp>
#include <autoppl/math/math.hpp>

#define PPL_WISHART_PARAM_SHAPE \
    "Wishart distribution scale matrix must have shape mat or selfadjmat " \
    "and n must be a scalar. "

namespace ppl {
namespace expr {
namespace dist {

/**
 * Wishart is a generic distribution expression representing
 * the wishart distribution.
 *
 * @tparam  VType       variable expression type for the scale matrix.
 * @tparam  NType       variable expression type for the scalar n.
 */

template <class VType
        , class NType>
struct Wishart: 
    util::DistExprBase<Wishart<VType, NType>>
{
private:
    using v_t = VType;
    using n_t = NType;

    static_assert(util::is_var_expr_v<v_t>);
    static_assert(util::is_var_expr_v<n_t>);
    static_assert(util::is_mat_v<v_t> &&
                  util::is_scl_v<n_t>,
                  PPL_DIST_SHAPE_MISMATCH
                  PPL_WISHART_PARAM_SHAPE
                  );

public:
    using value_t = util::cont_param_t;
    using base_t = util::DistExprBase<Wishart<v_t, n_t>>;
    using typename base_t::dist_value_t;

    Wishart(const v_t& v, 
            const n_t& n)
        : v_{v}, n_{n} 
    {}

    //template <class XType>
    //dist_value_t pdf(const XType& x) const 
    //{ 
    //    static_assert(util::is_var_v<XType>);
    //    static_assert(util::is_mat_v<XType>,
    //                  PPL_DIST_SHAPE_MISMATCH);
    //    return math::wishart_pdf(x.get(), v_.eval(), n_.eval());
    //}

    template <class XType>
    dist_value_t log_pdf(const XType& x) const 
    {
        static_assert(util::is_dist_assignable_v<XType>);
        static_assert(util::is_mat_v<XType>,
                      PPL_DIST_SHAPE_MISMATCH);
        return math::wishart_log_pdf(x.get(), v_.eval(), n_.eval());
    }
    
    template <class XType
            , class PtrPackType>
    auto ad_log_pdf(const XType& x,
                    const PtrPackType& pack) const
    {
        static_assert(util::is_dist_assignable_v<XType>);
        static_assert(util::is_mat_v<XType>,
                      PPL_DIST_SHAPE_MISMATCH);
        return ad::wishart_adj_log_pdf(x.ad(pack),
                                       v_.ad(pack),
                                       n_.ad(pack));
    }
        
    template <class PtrPackType>
    void bind(const PtrPackType& pack)
    { 
        static_cast<void>(pack);
        if constexpr (v_t::has_param) {
            v_.bind(pack);
        }
        if constexpr (n_t::has_param) {
            n_.bind(pack);
        }
    }

    void activate_refcnt() const 
    { 
        v_.activate_refcnt(); 
        n_.activate_refcnt();
    }

    template <class XType, class GenType>
    bool prune(XType&, GenType&) const { return false; }

private:
    v_t v_; 
    n_t n_;
};

} // namespace dist
} // namespace expr

/**
 * Builds a Wishart expression only when the parameters
 * are both valid continuous distribution parameter types.
 * See var_expr.hpp for more information.
 */
template <class VType, class NType
        , class = std::enable_if_t<
            util::is_valid_dist_param_v<VType> &&
            util::is_valid_dist_param_v<NType>
         > >
inline constexpr auto wishart(const VType& v_expr,
                              const NType& n_expr)
{
    using v_t = util::convert_to_param_t<VType>;
    using n_t = util::convert_to_param_t<NType>;

    v_t wrap_v_expr = v_expr;
    n_t wrap_n_expr = n_expr;

    return expr::dist::Wishart(wrap_v_expr, wrap_n_expr);
}

} // namespace ppl

#undef PPL_WISHART_PARAM_SHAPE
