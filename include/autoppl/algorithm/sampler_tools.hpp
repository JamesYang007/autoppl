#pragma once
#include <random>
#include <fastad>
#include <armadillo>
#include <autoppl/util/var_traits.hpp>

#define AUTOPPL_MH_UNKNOWN_VALUE_TYPE_ERROR \
    "Unknown value type: must be convertible to util::disc_param_t " \
    "such as uint64_t or util::cont_param_t such as double."

namespace ppl {
namespace alg {

/*
 * Returns number of parameters in the model.
 * Note that this assumes every parameter is univariate.
 */
template <class ModelType>
size_t get_n_params(const ModelType& model)
{
    size_t n = 0;
    auto get_n_params__ = [&](const auto& eq_node) {
        const auto& var = eq_node.get_variable();
        using var_t = std::decay_t<decltype(var)>;
        using state_t = typename util::var_traits<var_t>::state_t;
        n += (var.get_state() == state_t::parameter);
    };
    model.traverse(get_n_params__);
    return n;
}

/*
 * Initializes parameters with the given priors and
 * conditional distributions based on the model.
 * Random numbers are generated with gen.
 */
template <class ModelType, class GenType>
void init_params(ModelType& model, GenType& gen)
{
    // arbitrarily chosen radius for initial sampling
    constexpr double initial_radius = 5.;    

    auto init_params__ = [&](auto& eq_node) {
        auto& var = eq_node.get_variable();
        const auto& dist = eq_node.get_distribution();

        using var_t = std::decay_t<decltype(var)>;
        using value_t = typename util::var_traits<var_t>::value_t;
        using state_t = typename util::var_traits<var_t>::state_t;

        if (var.get_state() == state_t::parameter) {
            if constexpr (std::is_integral_v<value_t>) {
                std::uniform_int_distribution init_sampler(dist.min(), dist.max());
                var.set_value(init_sampler(gen));
            } else if constexpr (std::is_floating_point_v<value_t>) {
                std::uniform_real_distribution init_sampler(
                        std::max(dist.min(), -initial_radius), 
                        std::min(dist.max(), initial_radius)
                        );
                var.set_value(init_sampler(gen));
            } else {
                static_assert(!(std::is_integral_v<value_t> ||
                                std::is_floating_point_v<value_t>), 
                              AUTOPPL_MH_UNKNOWN_VALUE_TYPE_ERROR);
            }
        }
    };
    model.traverse(init_params__);
}

/*
 * Sets storage for values and adjoints for AD variables
 * in "vars" as respective elements in "values" and "adjoints".
 */
template <class ADVecType, class MatType>
void ad_bind_storage(ADVecType& vars, MatType& values, MatType& adjoints)
{
    auto values_it = values.begin();
    auto adjoints_it = adjoints.begin();
    std::for_each(vars.begin(), vars.end(), 
            [&](auto& var) {
                var.set_value_ptr(&(*values_it));
                var.set_adjoint_ptr(&(*adjoints_it));
                ++values_it;
                ++adjoints_it;
            });
}

/*
 * Compute Hamiltonian given potential and momentum vector.
 * Assumes that momentum vector was sampled from Normal(0, I).
 * @tparam  MatType matrix (vector) type supported by Armadillo.
 */
template <class MatType>
double hamiltonian(double potential, const MatType& r)
{
    return potential - 0.5 * arma::dot(r, r);
}

/*
 * Accepts or rejects with given probability using UniformDistType
 * object that works with GenType.
 * The uniform sampler must sample from [0,1].
 */
template <class UniformDistType, class GenType>
bool accept_or_reject(double p, 
                      UniformDistType&& unif_sampler,
                      GenType&& gen)
{
    double u = unif_sampler(gen);
    return (u <= p);
}

/*
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

/*
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
 * @param epsilon       step size
 * @param reuse_adj     flag to not compute gradient of L(theta) if
 *                      user can guarantee that theta_adj currently has it.
 *
 * @return  new potential energy (L(theta'))
 */
template <class ADExprType, class MatType>
double leapfrog(ADExprType& ad_expr,
                MatType& theta,
                MatType& theta_adj,
                MatType& r,
                double epsilon,
                bool reuse_adj)
{
    if (!reuse_adj) {
        reset_autodiff(ad_expr, theta_adj);
    }
    double half_step = epsilon/2.;
    r = r + half_step * theta_adj;
    theta = theta + epsilon * r;

    double new_potential = 
        reset_autodiff(ad_expr, theta_adj);
    r = r + half_step * theta_adj;

    return new_potential;
}

} // namespace alg
} // namespace ppl
