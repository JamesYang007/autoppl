#pragma once
#include <cstddef>
#include <cmath>
#include <Eigen/Dense>
#include <fastad_bits/util/shape_traits.hpp>
#include <fastad_bits/reverse/core/var_view.hpp>
#include <autoppl/expression/constraint/transformer.hpp>
#include <autoppl/util/value.hpp>
#include <autoppl/util/ad_boost/cov_inv_transform.hpp>

namespace ppl {
namespace expr {
namespace constraint {

struct PosDef {

    /**
     * Returns the number of unconstrained parameters based on
     * the rows (which is also cols) of a positive-definite matrix.
     */
    static constexpr size_t size(size_t rows)
    { return (rows * (rows + 1)) / 2; }

    /**
     * Transforms from constrained (c) to unconstrained (uc).
     */
    template <class CType, class UCType>
    static constexpr void transform(const CType& c,
                                    UCType& uc)
    {
        using value_t = typename CType::Scalar;
        using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;
        Eigen::LLT<mat_t> llt(c);
        mat_t lower = llt.matrixL();
        lower.diagonal().array() = lower.diagonal().array().log();

        size_t k = 0;
        for (int j = 0; j < lower.cols(); ++j) {
            for (int i = j; i < lower.rows(); ++i, ++k) {
                uc(k) = lower(i,j); 
            }
        }
    }

    /**
     * Inverse transforms from unconstrained parameters (uc),
     * which is vector-like in the sense that operator()(index) is defined,
     * to constrained parameter (c), which is matrix-like.
     * Lower should also be matrix-like supporting operator()(index, index)
     * which is used a temporary storage for the transformation.
     */
    template <class LowerType, class UCType, class CType>
    static constexpr void inv_transform(LowerType& lower,
                                        const UCType& uc,
                                        CType& c)
    { ad::boost::cov_inv_transform(lower, uc, c); }
};

// Specialization: Positive-Definite (matrix)
template <class ValueType>
struct Transformer<ValueType, mat, PosDef>
{
    using value_t = ValueType;
    using shape_t = mat;
    using var_t = util::var_t<value_t, shape_t>;
    using constraint_t = PosDef;
    using uc_view_t = ad::util::shape_to_raw_view_t<value_t, vec>;
    using view_t = ad::util::shape_to_raw_view_t<value_t, shape_t>;

    // only continuous value types can be constrained
    static_assert(util::is_cont_v<value_t>);

    /**
     * Constructs a Transformer object.
     * It represents a positive-definite matrix, and hence
     * ignores cols (second parameter) and treats rows as both rows and cols.
     *
     * @param   rows    number of constrained rows (and cols)
     */
    Transformer(size_t rows, 
                size_t,
                constraint_t=constraint_t())
        : uc_val_(nullptr, constraint_t::size(rows))
        , c_val_(nullptr, rows, rows)
        , lower_(nullptr, rows, rows)
        , v_val_(nullptr)
    {}

    void transform() {
        constraint_t::transform(c_val_, uc_val_);
    }

    /**
     * Inverse transforms from unconstrained parameters to constrained parameters.
     * Only the first visitor of the visit count will invoke the actual transformation.
     * The reference count is used to reset the visit count if 
     * the visit count has reached refcnt.
     *
     * @return  viewer of constrained parameters
     */
    void inv_transform(size_t refcnt) { 
        ++*v_val_;
        if (*v_val_ == 1) {
            constraint_t::inv_transform(lower_, uc_val_, c_val_);
        }
        *v_val_ = *v_val_ % refcnt;
    }

    /**
     * Inverse transform from unconstrained parameters to constrained parameters.
     * This should not have any memory dependency through calling bind.
     * It is expected that uc and c are vector-like 
     * in the sense that either row or column is 1.
     * In debug mode, we check that uc and c have correct sizes.
     *
     * @param   uc  vector of unconstrained parameters to transform
     * @param   c   vector of constrained parameters to populate
     */
    //template <class UCVecType, class CVecType>
    //void inv_transform(const UCVecType& uc,
    //                   CVecType& c) const
    //{
    //    assert(static_cast<size_t>(uc.size()) == size_uc());
    //    assert(static_cast<size_t>(c.size()) == size_c());
    //    assert(uc.rows() == 1 || uc.cols() == 1);
    //    assert(c.rows() == 1 || c.cols() == 1);

    //    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    //    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;
    //    mat_t uc_vec = uc;
    //    mat_t c_vec = c;
    //    mat_t lower(rows_c(), rows_c());
    //    lower.setZero();
    //    Eigen::Map<vec_t> uc_mp(uc_vec.data(), uc.size());
    //    Eigen::Map<mat_t> c_mp(c_vec.data(), rows_c(), rows_c());
    //    constraint_t::inv_transform(lower, uc_vec, c_mp);
    //    c = c_vec;
    //}

    /**
     * Creates an AD expression representing the inverse transform.
     * User must ensure that this gets called exactly refcnt number of times.
     *
     * @param   uc_val      beginning of unconstrained parameter values
     * @param   uc_adj      beginning of unconstrained parameter adjoints
     * @param   c_val       beginning of constrained value region.
     *                      The first size_c() elements will be used as temporary region
     *                      to compute a lower-triangular matrix.
     * @param   v_val       beginning of visit count
     * @param   refcnt      total reference count to determine when to loop visit count back to 0
     */
    template <class CurrPtrPack, class PtrPack>
    auto inv_transform_ad(const CurrPtrPack& curr_pack,
                          const PtrPack&,
                          size_t refcnt) const {
        ad::VarView<value_t, ad::vec> uc_view(curr_pack.uc_val, 
                                              curr_pack.uc_adj, 
                                              size_uc());
        return ad::boost::CovInvTransformNode(uc_view, 
                                              curr_pack.c_val,
                                              curr_pack.c_val + size_c(), 
                                              rows_c(), 
                                              curr_pack.v_val, 
                                              refcnt);
    }

    /**
     * Creates an AD expression representing the log-jacobian of inverse transform.
     * In general, this may need to reuse computed values from inverse transform.
     * User must guarantee that inverse transform AD expression that is bound to the same
     * resources as the return value of this function is evaluated before.
     */
    template <class CurrPtrPack, class PtrPack>
    auto logj_inv_transform_ad(const CurrPtrPack& curr_pack,
                               const PtrPack&) const {
        ad::VarView<value_t, ad::vec> uc_view(curr_pack.uc_val, 
                                              curr_pack.uc_adj, 
                                              size_uc());
        return ad::boost::LogJCovInvTransformNode(uc_view, rows_c());
    }

    /**
     * Initializes unconstrained values such that constrained (pos-def) matrix is identity.
     * This is equivalent to simply setting the unconstrained values to 0,
     * since diagonal is first exponentiated and the Cholesky decomposition of identity is identity.
     */
    template <class GenType, class ContDist>
    void init(GenType&, ContDist&) {
        uc_val_.setZero();
    }

    void activate_refcnt(size_t) const {}

    var_t& get_c() { return util::get(c_val_); }
    const var_t& get_c() const { return util::get(c_val_); }

    /**
     * Returns the dimension information for the viewers of unconstrained
     * and constrained parameters.
     */
    constexpr size_t size_uc() const { return uc_val_.size(); }
    constexpr size_t rows_uc() const { return uc_val_.rows(); }
    constexpr size_t cols_uc() const { return 1; }
    constexpr size_t size_c() const { return c_val_.size(); }
    constexpr size_t rows_c() const { return c_val_.rows(); }
    constexpr size_t cols_c() const { return c_val_.cols(); }

    /**
     * Returns the number of elements required to bind and compute 
     * unconstrained, constrained parameters and visit count.
     */
    constexpr size_t bind_size_uc() const { return size_uc(); }
    constexpr size_t bind_size_c() const { return 2*size_c(); }
    constexpr size_t bind_size_v() const { return 1; }

    /**
     * Binds unconstrained viewer to unconstrained region (viewed as a vector),
     * constrained viewer to constrained region (viewed as a matrix),
     * and internal visit count to visit count region.
     */
    template <class CurrPtrPack, class PtrPack>
    void bind(const CurrPtrPack& curr_pack,
              const PtrPack&) 
    {
        util::bind(uc_val_, curr_pack.uc_val, rows_uc(), cols_uc());
        util::bind(lower_, curr_pack.c_val, rows_c(), cols_c());
        util::bind(c_val_, curr_pack.c_val + size_c(), rows_c(), cols_c());
        util::bind(v_val_, curr_pack.v_val, 1, 1);
    }

private:
    uc_view_t uc_val_;
    view_t c_val_;
    view_t lower_;  // temporary lower-triangular matrix
    size_t* v_val_;
};

} // namespace constraint
} // namespace expr

constexpr inline auto pos_def()
{
    return expr::constraint::PosDef();
}

} // namespace ppl
