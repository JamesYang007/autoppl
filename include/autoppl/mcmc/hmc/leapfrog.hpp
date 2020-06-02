#pragma once
#include <fastad>
#include <armadillo>

namespace ppl {
namespace mcmc {

/**
 * Helper function for leapfrog algorithm.
 * Resets adjoints and then differentiates AD expression.
 * @param ad_expr   AD expression to differentiate.
 * @param adjoints  Armadillo generic matrix type that supports member fn "zeros".
 * @return  result of calling ad::autodiff on ad_expr.
 */
template <class ADExprType, class MatType>
double reset_autodiff(ADExprType& ad_expr, MatType& adjoints)
{
    // reset adjoints
    adjoints.zeros();
    // compute current gradient
    return ad::autodiff(ad_expr);
}

/**
 * Leapfrog algorithm.
 * Expects theta, theta_adj, r to be submatrix views of Armadillo matrix.
 * However, any matrix library supporting arithmetic
 * operations like +,-, * (scalar) should work.
 *
 * Updates theta, theta_adj, and r to contain the new leaped values
 * and adjoints.
 *
 * @param ad_expr       AD expression representing L(theta)
 *                      It must be built such that values are read from theta
 *                      and adjoints are placed into theta_adj.
 * @param theta         theta at which we want to start leaping
 * @param theta_adj     adjoint for theta. If not reusing, resets adjoints first.
 * @param r             momentum vector to start leaping
 * @param m_handler     momentum handler to compute correct dkinetic/dr
 * @param epsilon       step size
 * @param reuse_adj     flag to not compute gradient of L(theta) if
 *                      user can guarantee that theta_adj currently has it.
 *
 * @return  new potential energy (L(theta'))
 */
template <class ADExprType
        , class MatType
        , class MomentumHandlerType>
double leapfrog(ADExprType& ad_expr,
                MatType& theta,
                MatType& theta_adj,
                MatType& r,
                const MomentumHandlerType& m_handler,
                double epsilon,
                bool reuse_adj)
{
    if (!reuse_adj) {
        reset_autodiff(ad_expr, theta_adj);
    }
    const double half_step = epsilon/2.;
    r += half_step * theta_adj;

    theta += epsilon * m_handler.dkinetic_dr(r);

    const double new_potential = 
        -reset_autodiff(ad_expr, theta_adj);
    r += half_step * theta_adj;

    return new_potential;
}

} // namespace mcmc
} // namespace ppl
