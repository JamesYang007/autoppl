#pragma once
#include <cmath>
#include <autoppl/math/math.hpp>
#include <autoppl/util/traits/dist_expr_traits.hpp>
#include <fastad_bits/util/type_traits.hpp>
#include <Eigen/Dense>

// MSVC does not seem to support M_PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ppl {
namespace math {

using dist_value_t = util::dist_value_t;

/////////////////////////////////
// Compile-time Constants
/////////////////////////////////

inline constexpr double SQRT_TWO_PI = 
    2.506628274631000502415765284811045;
inline constexpr double LOG_SQRT_TWO_PI =
    0.918938533204672741780329736405617;

/////////////////////////////////
// Normal Density
/////////////////////////////////

// sss
template <class XType
        , class MeanType
        , class SigmaType
        , class = std::enable_if_t<
            std::is_arithmetic_v<XType> &&
            std::is_arithmetic_v<MeanType> &&
            std::is_arithmetic_v<SigmaType> 
        > >
inline dist_value_t normal_pdf(const XType& x, 
                               const MeanType& mean, 
                               const SigmaType& sigma)
{
    if (sigma <= 0) return math::neg_inf<dist_value_t>;
    dist_value_t z = (x - mean) / sigma;
    return std::exp(-0.5 * z * z) / (sigma * SQRT_TWO_PI);
}

// vss
template <class XType
        , class MeanType
        , class SigmaType
        , class = std::enable_if_t<
            std::is_arithmetic_v<MeanType> &&
            std::is_arithmetic_v<SigmaType> 
        > >
inline dist_value_t normal_pdf(const Eigen::MatrixBase<XType>& x, 
                               const MeanType& mean, 
                               const SigmaType& sigma)
{
    if (sigma <= 0) return math::neg_inf<dist_value_t>;
    dist_value_t z_sq = (x.array() - mean).matrix().squaredNorm() / (sigma * sigma);
    return std::exp(-0.5 * z_sq) / std::pow(sigma * SQRT_TWO_PI, x.size());
}

// vvs
template <class XType
        , class MeanType
        , class SigmaType
        , class = std::enable_if_t<
            std::is_arithmetic_v<SigmaType> 
        > >
inline dist_value_t normal_pdf(const Eigen::MatrixBase<XType>& x, 
                               const Eigen::MatrixBase<MeanType>& mean, 
                               const SigmaType& sigma)
{
    static_assert(ad::util::is_eigen_vector_v<MeanType>);
    assert(x.size() == mean.size());
    if (sigma <= 0) return math::neg_inf<dist_value_t>;
    dist_value_t z_sq = (x.array() - mean.array()).matrix().squaredNorm() / (sigma * sigma);
    return std::exp(-0.5 * z_sq) / std::pow(sigma * SQRT_TWO_PI, x.size());
}

// vsv, vsm
template <class XType
        , class MeanType
        , class SigmaType
        , class = std::enable_if_t<
            std::is_arithmetic_v<MeanType>
        > >
inline dist_value_t normal_pdf(const Eigen::MatrixBase<XType>& x, 
                               const MeanType& mean, 
                               const Eigen::MatrixBase<SigmaType>& sigma)
{
    if constexpr (ad::util::is_eigen_vector_v<SigmaType>) {
        assert(x.size() == sigma.size());
        if ((sigma.array() <= 0).any()) return math::neg_inf<dist_value_t>;
        dist_value_t z_sq = ((x.array() - mean)/sigma.array()).matrix().squaredNorm();
        return std::exp(-0.5 * z_sq) / (std::pow(SQRT_TWO_PI, x.size()) * sigma.array().prod());

    } else if constexpr (ad::util::is_eigen_matrix_v<SigmaType>) {
        assert(x.size() == sigma.rows());
        Eigen::LLT<Eigen::MatrixXd> llt(sigma);
        if (llt.info() != Eigen::Success) return math::neg_inf<dist_value_t>;
        dist_value_t z_sq = llt.matrixL().solve((x.array() - mean).matrix()).squaredNorm();
        dist_value_t det = llt.matrixL().determinant();
        return std::exp(-0.5 * z_sq) / (std::pow(SQRT_TWO_PI, x.size()) * det);
    }
}

// vvv, vvm
template <class XType
        , class MeanType
        , class SigmaType>
inline dist_value_t normal_pdf(const Eigen::MatrixBase<XType>& x, 
                               const Eigen::MatrixBase<MeanType>& mean, 
                               const Eigen::MatrixBase<SigmaType>& sigma)
{
    static_assert(ad::util::is_eigen_vector_v<MeanType>);

    if constexpr (ad::util::is_eigen_vector_v<SigmaType>) {
        assert(x.size() == sigma.size());
        assert(x.size() == mean.size());
        if ((sigma.array() <= 0).any()) return math::neg_inf<dist_value_t>;
        dist_value_t z_sq = ((x.array() - mean.array())/sigma.array()).matrix().squaredNorm();
        return std::exp(-0.5 * z_sq) / (std::pow(SQRT_TWO_PI, x.size()) * sigma.array().prod());
    }

    else if constexpr (ad::util::is_eigen_matrix_v<SigmaType>) {
        assert(x.size() == sigma.rows());
        assert(x.size() == mean.size());
        Eigen::LLT<Eigen::MatrixXd> llt(sigma);
        if (llt.info() != Eigen::Success) return math::neg_inf<dist_value_t>;
        dist_value_t z_sq = llt.matrixL().solve((x.array() - mean.array()).matrix()).squaredNorm();
        dist_value_t det = llt.matrixL().determinant();
        return std::exp(-0.5 * z_sq) / (std::pow(SQRT_TWO_PI, x.size()) * det);
    }
}

// sss
template <class XType
        , class MeanType
        , class SigmaType
        , class = std::enable_if_t<
            std::is_arithmetic_v<XType> &&
            std::is_arithmetic_v<MeanType> &&
            std::is_arithmetic_v<SigmaType> 
        > >
inline dist_value_t normal_log_pdf(const XType& x, 
                                   const MeanType& mean, 
                                   const SigmaType& sigma)
{
    if (sigma <= 0) return math::neg_inf<dist_value_t>;
    dist_value_t z = (x - mean) / sigma;
    return -0.5 * z * z - std::log(sigma) - LOG_SQRT_TWO_PI;
}

// vss
template <class XType
        , class MeanType
        , class SigmaType
        , class = std::enable_if_t<
            std::is_arithmetic_v<MeanType> &&
            std::is_arithmetic_v<SigmaType> 
        > >
inline dist_value_t normal_log_pdf(const Eigen::MatrixBase<XType>& x, 
                           const MeanType& mean, 
                           const SigmaType& sigma)
{
    if (sigma <= 0) return math::neg_inf<dist_value_t>;
    dist_value_t z_sq = (x.array() - mean).matrix().squaredNorm() / (sigma * sigma);
    return -0.5 * z_sq - (x.size() * (std::log(sigma) + LOG_SQRT_TWO_PI));
}

// vvs
template <class XType
        , class MeanType
        , class SigmaType
        , class = std::enable_if_t<
            std::is_arithmetic_v<SigmaType> 
        > >
inline dist_value_t normal_log_pdf(const Eigen::MatrixBase<XType>& x, 
                           const Eigen::MatrixBase<MeanType>& mean, 
                           const SigmaType& sigma)
{
    static_assert(ad::util::is_eigen_vector_v<MeanType>);
    assert(x.size() == mean.size());
    if (sigma <= 0) return math::neg_inf<dist_value_t>;
    dist_value_t z_sq = (x.array() - mean.array()).matrix().squaredNorm() / (sigma * sigma);
    return -0.5 * z_sq - (x.size() * (std::log(sigma) + LOG_SQRT_TWO_PI));
}

// vsv, vsm
template <class XType
        , class MeanType
        , class SigmaType
        , class = std::enable_if_t<
            std::is_arithmetic_v<MeanType>
        > >
inline dist_value_t normal_log_pdf(const Eigen::MatrixBase<XType>& x, 
                                   const MeanType& mean, 
                                   const Eigen::MatrixBase<SigmaType>& sigma)
{
    if constexpr (ad::util::is_eigen_vector_v<SigmaType>) {
        assert(x.size() == sigma.size());
        if ((sigma.array() <= 0).any()) return math::neg_inf<dist_value_t>;
        dist_value_t z_sq = ((x.array() - mean)/sigma.array()).matrix().squaredNorm();
        return -0.5 * z_sq - (x.size() * LOG_SQRT_TWO_PI) - std::log(sigma.array().prod());
    }

    else if constexpr (ad::util::is_eigen_matrix_v<SigmaType>) {
        assert(x.size() == sigma.rows());
        Eigen::LLT<Eigen::MatrixXd> llt(sigma);
        if (llt.info() != Eigen::Success) return math::neg_inf<dist_value_t>;
        dist_value_t z_sq = llt.matrixL().solve((x.array() - mean).matrix()).squaredNorm();
        dist_value_t det = llt.matrixL().determinant();
        return -0.5 * z_sq - (x.size() * LOG_SQRT_TWO_PI) - std::log(det);
    }
}

// vvv, vvm
template <class XType
        , class MeanType
        , class SigmaType>
inline dist_value_t normal_log_pdf(const Eigen::MatrixBase<XType>& x, 
                                   const Eigen::MatrixBase<MeanType>& mean, 
                                   const Eigen::MatrixBase<SigmaType>& sigma)
{
    if constexpr (ad::util::is_eigen_vector_v<SigmaType>) {
        assert(x.size() == sigma.size());
        assert(x.size() == mean.size());
        if ((sigma.array() <= 0).any()) return math::neg_inf<dist_value_t>;
        dist_value_t z_sq = ((x.array() - mean.array())/sigma.array()).matrix().squaredNorm();
        return -0.5 * z_sq - (x.size() * LOG_SQRT_TWO_PI) - std::log(sigma.array().prod());
    }

    else if constexpr (ad::util::is_eigen_matrix_v<SigmaType>) {
        assert(x.size() == sigma.rows());
        assert(x.size() == mean.size());
        Eigen::LLT<Eigen::MatrixXd> llt(sigma);
        if (llt.info() != Eigen::Success) return math::neg_inf<dist_value_t>;
        dist_value_t z_sq = llt.matrixL().solve((x.array() - mean.array()).matrix()).squaredNorm();
        dist_value_t det = llt.matrixL().determinant();
        return -0.5 * z_sq - (x.size() * LOG_SQRT_TWO_PI) - std::log(det);
    }
}

/////////////////////////////////
// Cauchy Density
/////////////////////////////////

// sss
template <class XType
        , class LocType
        , class ScaleType
        , class = std::enable_if_t<
            std::is_arithmetic_v<XType> &&
            std::is_arithmetic_v<LocType> &&
            std::is_arithmetic_v<ScaleType> 
        > >
inline dist_value_t cauchy_log_pdf(const XType& x, 
                                   const LocType& loc, 
                                   const ScaleType& scale)
{
    auto diff = x - loc;
    return (scale > 0) ? -std::log(scale + diff * diff / scale) : neg_inf<double>;
} 

// vss
template <class XType
        , class LocType
        , class ScaleType
        , class = std::enable_if_t<
            std::is_arithmetic_v<LocType> &&
            std::is_arithmetic_v<ScaleType> 
        > >
inline dist_value_t cauchy_log_pdf(const Eigen::MatrixBase<XType>& x, 
                                   const LocType& loc, 
                                   const ScaleType& scale)
{
    bool cond = scale > 0.;
    auto diff = x.array() - loc;
    return cond ? -(scale + (1./scale) * diff * diff).log().sum() : 
                  neg_inf<double>;
} 

// vvs
template <class XType
        , class LocType
        , class ScaleType
        , class = std::enable_if_t<
            std::is_arithmetic_v<ScaleType> 
        > >
inline dist_value_t cauchy_log_pdf(const Eigen::MatrixBase<XType>& x, 
                                    const Eigen::MatrixBase<LocType>& loc, 
                                    const ScaleType& scale)
{
    bool cond = scale > 0.;
    auto diff = x.array() - loc.array();
    return cond ? -(scale + (1./scale) * diff * diff).log().sum() : neg_inf<double>;
}

// vsv
template <class XType
        , class LocType
        , class ScaleType
        , class = std::enable_if_t<
            std::is_arithmetic_v<LocType> 
        > >
inline dist_value_t cauchy_log_pdf(const Eigen::MatrixBase<XType>& x, 
                                    const LocType& loc, 
                                    const Eigen::MatrixBase<ScaleType>& scale)
{
    bool cond = (scale.array() > 0.).all();
    auto diff = x.array() - loc;
    auto gamma = scale.array();
    return cond ? -(gamma + (1./gamma) * diff * diff).log().sum() : neg_inf<double>;
} 

// vvv
template <class XType
        , class LocType
        , class ScaleType>
inline dist_value_t cauchy_log_pdf(const Eigen::MatrixBase<XType>& x, 
                                    const Eigen::MatrixBase<LocType>& loc, 
                                    const Eigen::MatrixBase<ScaleType>& scale)
{
    bool cond = (scale.array() > 0.).all();
    auto diff = x.array() - loc.array();
    auto gamma = scale.array();
    return cond ? -(gamma + (1./gamma) * diff * diff).log().sum() : neg_inf<double>;
}

/////////////////////////////////
// Uniform Density
/////////////////////////////////

// sss
template <class XType
        , class MinType
        , class MaxType
        , class = std::enable_if_t<
            std::is_arithmetic_v<XType> &&
            std::is_arithmetic_v<MinType> &&
            std::is_arithmetic_v<MaxType> 
        > >
inline dist_value_t uniform_pdf(const XType& x, 
                                const MinType& min, 
                                const MaxType& max)
{
    return (min < x && x < max) ? 1. / (max - min) : 0;
} 

// vss
template <class XType
        , class MinType
        , class MaxType
        , class = std::enable_if_t<
            std::is_arithmetic_v<MinType> &&
            std::is_arithmetic_v<MaxType> 
        > >
inline dist_value_t uniform_pdf(const Eigen::MatrixBase<XType>& x, 
                                const MinType& min, 
                                const MaxType& max)
{
    bool cond = (min < x.array()).all() && (x.array() < max).all();
    return cond ? std::pow(1./(max-min), x.size()) : 0;
} 

// vvs
template <class XType
        , class MinType
        , class MaxType
        , class = std::enable_if_t<
            std::is_arithmetic_v<MaxType> 
        > >
inline dist_value_t uniform_pdf(const Eigen::MatrixBase<XType>& x, 
                                const Eigen::MatrixBase<MinType>& min, 
                                const MaxType& max)
{
    bool cond = (min.array() < x.array()).all() && (x.array() < max).all();
    return cond ? (1./(max-min.array())).prod() : 0;
} 

// vsv
template <class XType
        , class MinType
        , class MaxType
        , class = std::enable_if_t<
            std::is_arithmetic_v<MinType> 
        > >
inline dist_value_t uniform_pdf(const Eigen::MatrixBase<XType>& x, 
                                const MinType& min, 
                                const Eigen::MatrixBase<MaxType>& max)
{
    bool cond = (min < x.array()).all() && (x.array() < max.array()).all();
    return cond ? (1./(max.array()-min)).prod() : 0;
} 

// vvv
template <class XType
        , class MinType
        , class MaxType>
inline dist_value_t uniform_pdf(const Eigen::MatrixBase<XType>& x, 
                                const Eigen::MatrixBase<MinType>& min, 
                                const Eigen::MatrixBase<MaxType>& max)
{
    bool cond = (min.array() < x.array()).all() && (x.array() < max.array()).all();
    return cond ? (1./(max.array()-min.array())).prod() : 0;
}

// sss
template <class XType
        , class MinType
        , class MaxType
        , class = std::enable_if_t<
            std::is_arithmetic_v<XType> &&
            std::is_arithmetic_v<MinType> &&
            std::is_arithmetic_v<MaxType> 
        > >
inline dist_value_t uniform_log_pdf(const XType& x, 
                                    const MinType& min, 
                                    const MaxType& max)
{
    return (min < x && x < max) ? -std::log(max - min) : neg_inf<double>;
} 

// vss
template <class XType
        , class MinType
        , class MaxType
        , class = std::enable_if_t<
            std::is_arithmetic_v<MinType> &&
            std::is_arithmetic_v<MaxType> 
        > >
inline dist_value_t uniform_log_pdf(const Eigen::MatrixBase<XType>& x, 
                                    const MinType& min, 
                                    const MaxType& max)
{
    bool cond = (min < x.array()).all() && (x.array() < max).all();
    return cond ? 
        static_cast<dist_value_t>(x.size())*(-std::log(max-min)) : 
        neg_inf<double>;
} 

// vvs
template <class XType
        , class MinType
        , class MaxType
        , class = std::enable_if_t<
            std::is_arithmetic_v<MaxType> 
        > >
inline dist_value_t uniform_log_pdf(const Eigen::MatrixBase<XType>& x, 
                                    const Eigen::MatrixBase<MinType>& min, 
                                    const MaxType& max)
{
    bool cond = (min.array() < x.array()).all() && (x.array() < max).all();
    return cond ? -(max-min.array()).log().sum() : neg_inf<double>;
}

// vsv
template <class XType
        , class MinType
        , class MaxType
        , class = std::enable_if_t<
            std::is_arithmetic_v<MinType> 
        > >
inline dist_value_t uniform_log_pdf(const Eigen::MatrixBase<XType>& x, 
                                    const MinType& min, 
                                    const Eigen::MatrixBase<MaxType>& max)
{
    bool cond = (min < x.array()).all() && (x.array() < max.array()).all();
    return cond ? -(max.array()-min).log().sum() : neg_inf<double>;
} 

// vvv
template <class XType
        , class MinType
        , class MaxType>
inline dist_value_t uniform_log_pdf(const Eigen::MatrixBase<XType>& x, 
                                    const Eigen::MatrixBase<MinType>& min, 
                                    const Eigen::MatrixBase<MaxType>& max)
{
    bool cond = (min.array() < x.array()).all() && (x.array() < max.array()).all();
    return cond ? -(max.array()-min.array()).log().sum() : neg_inf<double>;
}

/////////////////////////////////
// Bernoulli Density
/////////////////////////////////

/**
 * Bernoulli pdf and log pdf (pmf actually).
 * It is defined to clip when p is out of the range [0,1],
 * i.e. if p < 0, then we take p = 0 and
 * if p > 1, then we take p = 1.
 */

// ss
template <class XType
        , class PType
        , class = std::enable_if_t<
            std::is_arithmetic_v<XType> &&
            std::is_arithmetic_v<PType>
        > >
inline dist_value_t bernoulli_pdf(const XType& x, 
                                  const PType& p)
{
    if (p <= 0) return (x == 0) + 0.;
    else if (p >= 1) return (x == 1) + 0.;

    if (x == 1) return p;
    else if (x == 0) return 1. - p;
    else return 0.0;
} 

// vs
template <class XType
        , class PType
        , class = std::enable_if_t<
            std::is_arithmetic_v<PType>
        > >
inline dist_value_t bernoulli_pdf(const Eigen::MatrixBase<XType>& x, 
                                  const PType& p)
{
    double pdf = 1.;
    for (int i = 0; i < x.size(); ++i) {
        pdf *= bernoulli_pdf(x(i), p);
    }
    return pdf;
} 

// vv
template <class XType
        , class PType>
inline dist_value_t bernoulli_pdf(const Eigen::MatrixBase<XType>& x, 
                                  const Eigen::MatrixBase<PType>& p)
{
    assert(x.size() == p.size());
    double pdf = 1.;
    for (int i = 0; i < x.size(); ++i) {
        pdf *= bernoulli_pdf(x(i), p(i));
    }
    return pdf;
}

// ss
template <class XType
        , class PType
        , class = std::enable_if_t<
            std::is_arithmetic_v<XType> &&
            std::is_arithmetic_v<PType>
        > >
inline dist_value_t bernoulli_log_pdf(const XType& x, 
                                      const PType& p)
{
    if (p <= 0) {
        if (x == 0) return 0.;
        else return neg_inf<PType>;
    }
    else if (p >= 1) {
        if (x == 1) return 0.;
        else return neg_inf<PType>;
    }

    if (x == 1) return std::log(p);
    else if (x == 0) return std::log(1. - p);
    else return neg_inf<PType>;
} 

// vs
template <class XType
        , class PType
        , class = std::enable_if_t<
            std::is_arithmetic_v<PType>
        > >
inline dist_value_t bernoulli_log_pdf(const Eigen::MatrixBase<XType>& x, 
                                      const PType& p)
{
    double logpdf = 0.;
    for (int i = 0; i < x.size(); ++i) {
        logpdf += bernoulli_log_pdf(x(i), p);
    }
    return logpdf;
} 

// vv
template <class XType
        , class PType>
inline dist_value_t bernoulli_log_pdf(const Eigen::MatrixBase<XType>& x, 
                                      const Eigen::MatrixBase<PType>& p)
{
    assert(x.size() == p.size());
    double logpdf = 0.;
    for (int i = 0; i < x.size(); ++i) {
        logpdf += bernoulli_log_pdf(x(i), p(i));
    }
    return logpdf;
}

/////////////////////////////////
// Wishart Density
// Note: drops unnecessary constant terms
/////////////////////////////////
template <class XType
        , class VType
        , class NType>
inline dist_value_t wishart_log_pdf(const Eigen::MatrixBase<XType>& x,
                                    const Eigen::MatrixBase<VType>& v,
                                    const NType& n)
{
    Eigen::LLT<Eigen::MatrixXd> x_llt(x);
    Eigen::LLT<Eigen::MatrixXd> v_llt(v);
    auto log_det_x = std::log(x_llt.matrixL().determinant());
    auto log_det_v = std::log(v_llt.matrixL().determinant());
    auto tr = v_llt.solve(x).trace();
    dist_value_t p = v.rows();
    return (n - p - 1.) * log_det_x - 0.5 * tr - n * log_det_v;
}

} // namespace math
} // namespace ppl
