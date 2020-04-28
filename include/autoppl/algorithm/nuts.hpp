#pragma once
#include <cassert>
#include <cmath>
#include <stack>
#include <optional>
#include <type_traits>
#include <armadillo>
#include <fastad>
#include <autoppl/util/var_traits.hpp>
#include <autoppl/expression/model/glue_node.hpp>
#include <autoppl/expression/model/eq_node.hpp>
#include <autoppl/algorithm/sampler_tools.hpp>

namespace ppl {
namespace alg {

/* 
 * Struct to pack input to calling build_tree.
 */
template <class ADExprType
        , class SubviewType
        >
struct TreeInput
{
    using ad_expr_t = ADExprType;
    using subview_t = SubviewType;

    using ad_expr_ref_t = std::reference_wrapper<ad_expr_t>;
    using subview_ref_t = std::reference_wrapper<subview_t>;

    TreeInput(ad_expr_t& ad_expr,
              subview_t& theta,
              subview_t& theta_adj,
              subview_t& rho,
              double log_u,
              int8_t v,
              double epsilon,
              double ham
               )
        : ad_expr_ref{ad_expr} 
        , theta_ref{theta}
        , theta_adj_ref{theta_adj}
        , rho_ref{rho}
        , log_u{log_u}
        , v{v}
        , epsilon{epsilon}
        , ham{ham}
    {}

    ad_expr_ref_t ad_expr_ref;
    subview_ref_t theta_ref;
    subview_ref_t theta_adj_ref;
    subview_ref_t rho_ref;
    double log_u;
    int8_t v;
    double epsilon;
    double ham;
};

/* 
 * Struct to pack output from calling build_tree.
 * If the optional references are not set, then build_tree
 * won't overwrite the content. Otherwise, overwrites.
 */

template <class SubviewType>
struct TreeOutput
{
    using subview_t = SubviewType;
    using subview_ref_t = std::reference_wrapper<subview_t>;
    using opt_subview_ref_t = std::optional<
        std::reference_wrapper<subview_t>>;

    TreeOutput(subview_t& theta_prime)
        : theta_prime_ref{theta_prime}
    {}

    subview_ref_t theta_prime_ref; 
    opt_subview_ref_t opt_theta_ref;
    opt_subview_ref_t opt_rho_ref;
    size_t n = 0;
    bool s = false;
    double alpha = 0.;
    size_t n_alpha = 0;
    double potential = 0.;
};

/*
 * Copies only values: n, s, alpha, n_alpha, potential.
 */
template <class SubviewType>
void tree_output_copy_values(TreeOutput<SubviewType>& dest,
                             const TreeOutput<SubviewType>& src)
{
    dest.n = src.n;
    dest.s = src.s;
    dest.alpha = src.alpha;
    dest.n_alpha = src.n_alpha;
    dest.potential = src.potential;
}

/*
 * Checks if NUTS should stop the doubling process for current
 * subtree with leftmost (minus) and rightmost (plus) states
 * if given the current momentum for each state, an infinitesmal
 * change would cause the distance between the position vectors
 * will decrease.
 *
 * @tparam  MatType     Armadillo generic matrix (vector) type.
 */
template <class MatType>
bool check_entropy(const MatType& theta_plus,
                   const MatType& theta_minus,
                   const MatType& rho_plus,
                   const MatType& rho_minus)
{
    auto diff = theta_plus - theta_minus;
    return (arma::dot(diff, rho_plus) >= 0) &&
           (arma::dot(diff, rho_minus) >= 0);
}

/* 
 * Building binary tree for sampling candidates.
 * Helper function to obtain the forward/backward-most position and momentum.
 * Accept/reject policy is based on UniformDistType parameter and GenType
 * By default, uses standard library uniform_real_distribution on (0.,1.).
 * By default, GenType usees standard library mt19937.
 *
 * Note that the caller MUST have input theta_adj already pre-computed.
 */
template <size_t n_params
        , class InputType
        , class OutputType
        , class UniformDistType = std::uniform_real_distribution<double>
        , class GenType = std::mt19937
    >
void build_tree(InputType& input, 
                OutputType& output,
                uint8_t depth,
                UniformDistType unif_sampler = UniformDistType(0., 1.),
                GenType gen = GenType()
                )
{
    constexpr double delta_max = 1000;  // suggested by Gelman

    // base case
    if (depth == 0) {
        double new_potential = leapfrog(input.ad_expr_ref.get(),
                                        input.theta_ref.get(),
                                        input.theta_adj_ref.get(),
                                        input.rho_ref.get(),
                                        input.v * input.epsilon,
                                        true // always reuse previous adjoint
                                        );
        double new_ham = hamiltonian(new_potential, input.rho_ref.get());
        
        // if other optional pm in output is set, copy there as well
        // Note: opt_theta_ref has value iff opt_rho_ref has value
        if (output.opt_theta_ref.has_value()) {
            output.opt_theta_ref->get() = input.theta_ref.get(); 
            output.opt_rho_ref->get() = input.rho_ref.get(); 
        }

        // always copy into theta_prime 
        output.theta_prime_ref.get() = input.theta_ref.get();

        output.n = (input.log_u <= new_ham);
        output.s = (input.log_u < delta_max + new_ham);
        output.alpha = std::min(1., std::exp(new_ham - input.ham));
        output.n_alpha = 1;
        output.potential = new_potential;

        return;
    }

    // recursion
    arma::mat pm(0, 2);
    OutputType first_output = output;    // first recursive output

    // if optional theta (and rho) are not set,
    // resize pm for usage.
    if (!first_output.opt_theta_ref.has_value()) {
        pm.resize(n_params, 2);
    }

    auto theta = pm.unsafe_col(0);
    auto rho = pm.unsafe_col(1);

    // if pm being used, bind to first output
    if (!first_output.opt_theta_ref.has_value()) {
        first_output.opt_theta_ref = theta;
        first_output.opt_rho_ref = rho;
    }
    
    build_tree<n_params>(input, first_output, depth - 1);

    // ham way below threshold of delta_max: early finish
    // simply copy first output's values into caller's output.
    if (!first_output.s) {
        tree_output_copy_values(output, first_output);
        return;
    }

    // second recursion with same input from original caller.
    // This time, we don't have any other pm to update.
    // Need a new theta_prime storage though.
    arma::mat::fixed<n_params, 1> theta_tmp;
    auto theta_double_prime = theta_tmp.unsafe_col(0);
    OutputType second_output = output;
    second_output.opt_theta_ref.reset();
    second_output.opt_rho_ref.reset();
    second_output.theta_prime_ref = theta_double_prime;

    build_tree<n_params>(input, second_output, depth - 1);

    // accept with n''/(n' + n'') probability
    // if accepting, also copy over potential from second output
    // otherwise, only copy potential from FIRST output
    if (second_output.n) {
        double accept_prob = second_output.n / (first_output.n + second_output.n);
        bool accept = accept_or_reject(accept_prob, unif_sampler, gen);
        if (accept) { 
            first_output.theta_prime_ref.get() = 
                second_output.theta_prime_ref.get();
            output.potential = second_output.potential;
        } else {
            output.potential = first_output.potential;
        }
    } else {    // if second output n were 0, no need to sample: always reject
        output.potential = first_output.potential;
    }

    // check entropy with current backward/forward-most states
    // note that input's theta is backwardmost iff v == -1.
    bool is_entroping = false;
    if (input.v == -1) {
        is_entroping = check_entropy(
                first_output.opt_theta_ref->get(),  // theta_plus
                input.theta_ref.get(),              // theta_minus
                first_output.opt_rho_ref->get(),    // rho_plus
                input.rho_ref.get()                 // rho_minus
                );
    } else {
        is_entroping = check_entropy(
                input.theta_ref.get(),              // theta_plus
                first_output.opt_theta_ref->get(),  // theta_minus
                input.rho_ref.get(),                // rho_plus
                first_output.opt_rho_ref->get()     // rho_minus
                );
    }

    output.n = first_output.n + second_output.n;
    output.s = second_output.s && is_entroping;
    output.alpha = first_output.alpha + second_output.alpha;
    output.n_alpha = first_output.n_alpha + second_output.n_alpha;
}

/*
 * Finds a reasonable epsilon for NUTS algorithm.
 * @param   ad_expr     AD expression bound to theta and theta_adj
 */
template <size_t n_params
        , class ADExprType
        , class MatType>
double find_reasonable_epsilon(ADExprType& ad_expr,
                               MatType& theta,
                               MatType& theta_adj)
{
    double eps = 1.;
    const double diff_bound = -std::log(2);

    arma::mat::fixed<n_params, 2> r_mat;
    auto r = r_mat.unsafe_col(0);
    auto r_orig = r_mat.unsafe_col(1);

    arma::mat::fixed<n_params, 2> theta_mat;
    auto theta_orig = theta_mat.unsafe_col(0);
    auto theta_adj_orig = theta_mat.unsafe_col(1);

    // initialize r ~ N(0, I)
    r.randn();

    // differentiate first to get adjoints and potential
    const double potential_orig = ad::autodiff(ad_expr); 
    const double ham_orig = hamiltonian(potential_orig, r);

    // save original value and adjoint
    theta_orig = theta;
    theta_adj_orig = theta_adj;
    r_orig = r;
    
    double potential_curr = leapfrog(
            ad_expr, theta, theta_adj, r, eps, true);
    double ham_curr = hamiltonian(potential_curr, r);

    int a = 2*(ham_curr - ham_orig > diff_bound) - 1;

    while ((a * (ham_curr - ham_orig) > a * diff_bound)) {

        eps *= std::pow(2, a);

        // copy back original value and adjoint
        theta = theta_orig;
        theta_adj = theta_adj_orig;
        r = r_orig;

        potential_curr = leapfrog(
                ad_expr, theta, theta_adj, r, eps, true);
        ham_curr = hamiltonian(potential_curr, r);
    }

    // copy back original value and adjoint
    theta = theta_orig;
    theta_adj = theta_adj_orig;

    return eps;
}

} // namespace alg

/*
 * No-U-Turn Sampler (NUTS)
 *
 * The initialization is performed by sampling from the 
 * prior and conditional distributions specified by the model.
 */
template <class ModelType>
void nuts(ModelType& model,
          size_t warmup,
          size_t n_samples,
          size_t n_adapt,
          size_t seed = 0,
          size_t max_depth = 10,
          double delta = 0.6
          )
{

    // initialization of meta-variables
    double log_eps_bar = 0.;
    double H_bar = 0;
    constexpr double gamma = 0.05;
    constexpr double t0 = 10;
    constexpr double kappa = 0.75;
    constexpr size_t n_params = get_n_params_v<ModelType>;
    std::mt19937 gen(seed);
    std::uniform_int_distribution direction_sampler(0, 1);
    std::uniform_real_distribution metrop_sampler(0., 1.);

    // keys needed to construct a correct AD expression from model
    // key: address of original variable tags
    std::vector<const void*> keys(n_params, nullptr);
    auto keys_it = keys.begin();
    auto get_keys = [=](auto& eq_node) mutable {
        auto& var = eq_node.get_variable();
        using var_t = std::decay_t<decltype(var)>;
        if constexpr (util::is_param_v<var_t>) {
            *keys_it = &var;
            ++keys_it;
        }
    };
    model.traverse(get_keys);

    // momentum matrix
    constexpr uint8_t n_rhos_cached = 2;
    arma::mat::fixed<n_params, n_rhos_cached> rho_mat;
    auto rho_minus = rho_mat.unsafe_col(0);
    auto rho_plus = rho_mat.unsafe_col(1);

    // position matrix for thetas and adjoints
    constexpr uint8_t n_thetas_cached = 7;
    arma::mat::fixed<n_params, n_thetas_cached> theta_mat;
    auto theta_minus = theta_mat.unsafe_col(0);
    auto theta_minus_adj = theta_mat.unsafe_col(1);
    auto theta_plus = theta_mat.unsafe_col(2);
    auto theta_plus_adj = theta_mat.unsafe_col(3);
    auto theta_curr = theta_mat.unsafe_col(4);
    auto theta_curr_adj = theta_mat.unsafe_col(5);
    auto theta_prime = theta_mat.unsafe_col(6);

    // TODO: remove with Jacob's thing
    std::vector<ad::Var<double>> theta_minus_ad(n_params);
    std::vector<ad::Var<double>> theta_plus_ad(n_params);
    std::vector<ad::Var<double>> theta_curr_ad(n_params);
    alg::ad_bind_storage(theta_minus_ad, theta_minus, theta_minus_adj);
    alg::ad_bind_storage(theta_plus_ad, theta_plus, theta_plus_adj);
    alg::ad_bind_storage(theta_curr_ad, theta_curr, theta_curr_adj);

    // AD Expressions for theta_minus and theta_plus.
    // Note that these expressions are the only ones used ever.
    // L(theta) (potential of input theta)
    auto theta_minus_ad_expr = model.ad_log_pdf(keys, theta_minus_ad);
    auto theta_plus_ad_expr = model.ad_log_pdf(keys, theta_plus_ad);
    auto theta_curr_ad_expr = model.ad_log_pdf(keys, theta_curr_ad);
    
    // initialize model tags using model specs
    // copies the initialized values into theta_curr
    alg::init_params(model, gen);    
    auto theta_curr_it = theta_curr.begin();
    auto copy_params_potential = [=](const auto& eq_node) mutable {
        const auto& var = eq_node.get_variable();
        using var_t = std::decay_t<decltype(var)>;
        if constexpr (util::is_param_v<var_t>) {
            *theta_curr_it = var.get_value(); 
            ++theta_curr_it;
        }
    };
    model.traverse(copy_params_potential);

    // initialize current potential (will be "previous" starting in for-loop)
    double potential_prev = ad::evaluate(theta_curr_ad_expr);

    // initialize rest of the metavariables
    double log_eps = std::log(alg::find_reasonable_epsilon<n_params>(
            theta_curr_ad_expr, theta_curr, theta_curr_adj)); 
    const double mu = std::log(10.) + log_eps;

    // tree output struct type
    using subview_t = std::decay_t<decltype(rho_minus)>;
    using tree_output_t = alg::TreeOutput<subview_t>;

    for (size_t i = 0; i < n_samples + warmup; ++i) {

        // re-initialize vectors
        theta_plus = theta_minus = theta_curr;
        alg::reset_autodiff(theta_minus_ad_expr, theta_minus_adj); 
        theta_plus_adj = theta_minus_adj;   // no need to differentiate again
        size_t j = 0;
        size_t n = 1;
        bool s = true;

        // rho0 ~ N(0, I) 
        // optimization: rho0 will be assigned to rho_minus and rho_plus afterwards
        rho_minus.randn(); 
        rho_plus = rho_minus;

        // u ~ Uniform[0, exp(L(theta^{m-1}) - 0.5 * r0^2)] 
        const double ham_prev = alg::hamiltonian(potential_prev, rho_minus);
        std::uniform_real_distribution unif_sampler(0., std::exp(ham_prev));
        double log_u = std::log(unif_sampler(gen));

        tree_output_t output(theta_prime);

        while (s && (j < max_depth)) {
            int8_t v = 2 * direction_sampler(gen) - 1; // -1 or 1
            if (v == -1) {
                auto input = alg::TreeInput(
                    theta_minus_ad_expr, theta_minus, theta_minus_adj, rho_minus,
                    log_u, v, std::exp(log_eps), ham_prev
                );
                output.opt_theta_ref.reset();
                output.opt_rho_ref.reset();
                alg::build_tree<n_params>(input, output, j, metrop_sampler, gen);
            } else {
                auto input = alg::TreeInput(
                    theta_plus_ad_expr, theta_plus, theta_plus_adj, rho_plus,
                    log_u, v, std::exp(log_eps), ham_prev
                );
                output.opt_theta_ref.reset();
                output.opt_rho_ref.reset();
                alg::build_tree<n_params>(input, output, j, metrop_sampler, gen);
            }

            if (output.s) {
                double p = output.n/static_cast<double>(n);
                // if accepted
                if (alg::accept_or_reject(p, metrop_sampler, gen)) {
                    theta_curr = theta_prime;
                    potential_prev = output.potential;
                }
            }

            n += output.n;
            s = output.s && alg::check_entropy(
                    theta_plus, theta_minus, rho_plus, rho_minus); 
            ++j;
        } // end while
        
        // Epsilon Dual Averaging
        if (i < n_adapt) {
            double adapt_ratio = 1./(i + 1 + t0);
            H_bar = (1 - adapt_ratio) * H_bar + 
                    adapt_ratio * (delta - output.alpha/output.n_alpha);
            log_eps = mu - std::sqrt(i+1)/gamma * H_bar;
            double m_ratio = std::pow(i+1, -kappa);
            log_eps_bar = m_ratio * log_eps +
                          (1 - m_ratio) * log_eps_bar;
        } // end if

        // store sample theta_curr only after burning
        if (i >= warmup) {
            auto theta_curr_it = theta_curr.begin();
            auto store_sample = [=](auto& eq_node) mutable {
                auto& var = eq_node.get_variable();
                using var_t = std::decay_t<decltype(var)>;
                if constexpr (util::is_param_v<var_t>) {
                    auto storage_ptr = var.get_storage();
                    storage_ptr[i - warmup] = *theta_curr_it;
                    ++theta_curr_it;
                }
            };
            model.traverse(store_sample);
        }

    } // end for
}

} // namespace ppl
