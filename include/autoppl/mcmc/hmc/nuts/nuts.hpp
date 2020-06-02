#pragma once
#include <chrono>
#include <cassert>
#include <cmath>
#include <stack>
#include <optional>
#include <type_traits>
#include <iostream>
#include <armadillo>
#include <fastad>
#include <autoppl/util/var_traits.hpp>
#include <autoppl/util/logging.hpp>
#include <autoppl/mcmc/hmc/nuts/tree_utils.hpp>
#include <autoppl/expression/model/glue_node.hpp>
#include <autoppl/expression/model/eq_node.hpp>
#include <autoppl/math/smoothers.hpp>
#include <autoppl/mcmc/sampler_tools.hpp>
#include <autoppl/mcmc/hmc/ad_utils.hpp>
#include <autoppl/mcmc/hmc/leapfrog.hpp>
#include <autoppl/mcmc/hmc/hamiltonian.hpp>
#include <autoppl/mcmc/hmc/nuts/configs.hpp>

namespace ppl {
namespace mcmc {

/**
 * Checks if state is entroping based on integrated momentum vector
 * across the path and the scaled momentum at the ends of the path.
 *
 * @param   rho             integrated momentum vector
 * @param   p_beg_scaled    scaled momentum beginning of 
 *                          current path (given the direction)
 * @param   p_end_scaled    scaled momentum end of 
 *                          current path (given the direction).
 */
template <class MatType1, class MatType2, class MatType3>
bool check_entropy(const MatType1& rho, 
                   const MatType2& p_beg_scaled,
                   const MatType3& p_end_scaled)
{
    return arma::dot(rho, p_beg_scaled) > 0 &&
           arma::dot(rho, p_end_scaled) > 0;
}

/**
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
        , class UniformDistType
        , class GenType
        , class MomentumHandlerType
    >
TreeOutput build_tree(InputType& input, 
                      uint8_t depth,
                      UniformDistType& unif_sampler,
                      GenType& gen,
                      const MomentumHandlerType& momentum_handler
                      )
{
    constexpr double delta_max = 1000;  // suggested by Gelman

    // base case
    if (depth == 0) {
        double new_potential = leapfrog(input.ad_expr_ref.get(),
                                        input.theta_ref.get(),
                                        input.theta_adj_ref.get(),
                                        input.p_most_ref.get(),
                                        momentum_handler,
                                        input.v * input.epsilon,
                                        true // always reuse previous adjoint
                                        );
        double new_kinetic = momentum_handler.kinetic(input.p_most_ref.get());
        double new_ham = hamiltonian(new_potential, new_kinetic);

        // update number of leapfrogs
        ++(input.n_leapfrog_ref.get());

        // update LSE of weights
        input.log_sum_weight_ref.get() = math::lse(
                input.log_sum_weight_ref.get(), 
                input.ham - new_ham);

        // update sum of probabilities
        input.sum_metro_prob_ref.get() += (input.ham - new_ham > 0) ? 
                1 : std::exp(input.ham - new_ham);

        // always copy into theta_prime 
        input.theta_prime_ref.get() = input.theta_ref.get();

        // update momenta of beginning of subtree (moving in the direction of input.v)
        input.p_beg_ref.get() = input.p_most_ref.get();
        input.p_beg_scaled_ref.get() = 
            momentum_handler.dkinetic_dr(input.p_most_ref.get());

        // update momenta of end of subtree (moving in the direction of input.v)
        input.p_end_ref.get() = input.p_beg_ref.get();
        input.p_end_scaled_ref.get() = input.p_beg_scaled_ref.get();

        // update integrated momentum
        input.rho_ref.get() += input.p_most_ref.get();

        // return validity and new potential 
        return TreeOutput(
                (new_ham - input.ham <= delta_max),
                new_potential
            );
    }

    // recursion
    arma::mat::fixed<n_params, 3> mat_first(arma::fill::zeros);
    auto p_end_inner = mat_first.col(0);
    auto p_end_scaled_inner = mat_first.col(1);
    auto rho_first = mat_first.col(2);
    double log_sum_weight_first = -std::numeric_limits<double>::infinity();

    // create a new input for first recursion
    // some references have to rebound
    InputType first_input = input; 
    first_input.p_end_ref = p_end_inner;
    first_input.p_end_scaled_ref = p_end_scaled_inner;
    first_input.rho_ref = rho_first;
    first_input.log_sum_weight_ref = log_sum_weight_first;

    // build first subtree
    TreeOutput first_output = 
        build_tree<n_params>(first_input, depth - 1, 
                             unif_sampler, gen, momentum_handler);

    // if first subtree is already invalid, early exit
    // note that caller will break out of doubling process now,
    // so we do not have to worry about updating the other momentum vectors
    if (!first_output.valid) { return first_output; }

    // second recursion
    arma::mat::fixed<n_params, 4> mat_second(arma::fill::zeros);
    auto theta_double_prime = mat_second.col(0);
    auto p_beg_inner = mat_second.col(1);
    auto p_beg_scaled_inner = mat_second.col(2);
    auto rho_second = mat_second.col(3);
    double log_sum_weight_second = -std::numeric_limits<double>::infinity();

    // create a new input for second recursion
    InputType second_input = input;
    second_input.theta_prime_ref = theta_double_prime;
    second_input.p_beg_ref = p_beg_inner;
    second_input.p_beg_scaled_ref = p_beg_scaled_inner;
    second_input.rho_ref = rho_second;
    second_input.log_sum_weight_ref = log_sum_weight_second;

    // build second subtree
    TreeOutput second_output = 
        build_tree<n_params>(second_input, depth - 1, 
                             unif_sampler, gen, momentum_handler);

    // if second subtree is invalid, early exit
    // note that we must return first output since it has the potential
    // of the first proposal and we ignore the second proposal
    if (!second_output.valid) { 
        first_output.valid = false;
        return first_output; 
    }

    // create output to return at the end
    TreeOutput output;

    // sample proposal and update corresponding potential
    double log_sum_weight_curr = math::lse(
            log_sum_weight_first, log_sum_weight_second
            );
    input.log_sum_weight_ref.get() = math::lse(
            input.log_sum_weight_ref.get(), log_sum_weight_curr
            );

    // note: accept_prob is mathematically guaranteed to be <= 1
    double accept_prob = std::exp(log_sum_weight_second - log_sum_weight_curr);
    bool accept = accept_or_reject(accept_prob, unif_sampler, gen);
    if (accept) { 
        input.theta_prime_ref.get() = 
            second_input.theta_prime_ref.get();
        output.potential = second_output.potential;
    } else {
        output.potential = first_output.potential;
    }

    // check if current subtree is still valid based
    // on entropy condition
    auto rho_curr = rho_first + rho_second;
    input.rho_ref.get() += rho_curr;
    output.valid =
        check_entropy(rho_curr,
                      input.p_beg_scaled_ref.get(),
                      input.p_end_scaled_ref.get()) &&
        check_entropy(rho_first + p_beg_inner, 
                      input.p_beg_scaled_ref.get(), 
                      p_beg_scaled_inner) &&
        check_entropy(p_end_inner + rho_second, 
                      p_end_scaled_inner, 
                      input.p_end_scaled_ref.get());

    return output;
}

/**
 * Finds a reasonable epsilon for NUTS algorithm.
 * @param   ad_expr     AD expression bound to theta and theta_adj
 */
template <size_t n_params
        , class ADExprType
        , class MatType
        , class MomentumHandlerType>
double find_reasonable_epsilon(double eps,
                               ADExprType& ad_expr,
                               MatType& theta,
                               MatType& theta_adj,
                               const MomentumHandlerType& momentum_handler)
{
    // See (STAN) for reference: if epsilon is way out of bounds, just return eps
    if (eps == 0 || eps > 1e7) return eps;

    const double diff_bound = std::log(0.8);

    arma::mat::fixed<n_params, 1> r_mat(arma::fill::zeros);
    auto r = r_mat.col(0);

    arma::mat::fixed<n_params, 2> theta_mat(arma::fill::zeros);
    auto theta_orig = theta_mat.col(0);
    auto theta_adj_orig = theta_mat.col(1);

    // sample momentum vector based on handler
    momentum_handler.sample(r);

    // differentiate first to get adjoints and hamiltonian
    const double potential_orig = -ad::autodiff(ad_expr); 
    double kinetic_orig = momentum_handler.kinetic(r);
    double ham_orig = hamiltonian(potential_orig, kinetic_orig);

    // save original value and adjoint
    theta_orig = theta;
    theta_adj_orig = theta_adj;
    
    // get current hamiltonian after leapfrog
    double potential_curr = leapfrog(
            ad_expr, theta, theta_adj, r, momentum_handler, eps, true);
    double kinetic_curr = momentum_handler.kinetic(r);
    double ham_curr = hamiltonian(potential_curr, kinetic_curr);

    int a = (ham_orig - ham_curr > diff_bound) ? 1 : -1;

    while (1) {

        // check if break condition holds
        if ( ((a == 1) && !(ham_orig - ham_curr > diff_bound)) || 
             ((a == -1) && !(ham_orig - ham_curr < diff_bound)) ) {
            break;
        }

        // update epsilon
        eps *= (a == -1) ? 0.5 : 2;

        // copy back original value and adjoint
        theta = theta_orig;
        theta_adj = theta_adj_orig;

        // recompute original hamiltonian with new momentum
        momentum_handler.sample(r);
        kinetic_orig = momentum_handler.kinetic(r);
        ham_orig = hamiltonian(potential_orig, kinetic_orig);

        // leapfrog and compute current hamiltonian
        potential_curr = leapfrog(
                ad_expr, theta, theta_adj, r, momentum_handler, eps, true);
        kinetic_curr = momentum_handler.kinetic(r);
        ham_curr = hamiltonian(potential_curr, kinetic_curr);

    }

    // copy back original value and adjoint
    theta = theta_orig;
    theta_adj = theta_adj_orig;

    return eps;
}

} // namespace mcmc

/**
 * No-U-Turn Sampler (NUTS)
 */
template <class ModelType
        , class NUTSConfigType = NUTSConfig<>>
void nuts(ModelType& model, NUTSConfigType config = NUTSConfigType())
{
    // initialization of meta-variables
    constexpr size_t n_params = get_n_params_v<ModelType>;
    std::mt19937 gen(config.seed);
    std::uniform_int_distribution direction_sampler(0, 1);
    std::uniform_real_distribution unif_sampler(0., 1.);

    // momentum matrix (for stability reasons we require knowing 4 momentum)
    // left-subtree backwardmost momentum => bb
    // left-subtree forwardmost momentum => bf
    // right-subtree backwardmost momentum => fb
    // right-subtree forwardmost momentum => ff
    // scaled versions are based on hamiltonian adjusted covariance matrix
    constexpr uint8_t n_p_cached = 8;
    arma::mat::fixed<n_params, n_p_cached> p_mat(arma::fill::zeros);
    auto p_bb = p_mat.col(0);
    auto p_bb_scaled = p_mat.col(1);
    auto p_bf = p_mat.col(2);
    auto p_bf_scaled = p_mat.col(3);
    auto p_fb = p_mat.col(4);
    auto p_fb_scaled = p_mat.col(5);
    auto p_ff = p_mat.col(6);
    auto p_ff_scaled = p_mat.col(7);

    // position matrix for thetas and adjoints
    constexpr uint8_t n_thetas_cached = 7;
    arma::mat::fixed<n_params, n_thetas_cached> theta_mat(arma::fill::zeros);
    auto theta_bb = theta_mat.col(0);
    auto theta_bb_adj = theta_mat.col(1);
    auto theta_ff = theta_mat.col(2);
    auto theta_ff_adj = theta_mat.col(3);
    auto theta_curr = theta_mat.col(4);
    auto theta_curr_adj = theta_mat.col(5);
    auto theta_prime = theta_mat.col(6);

    // integrated momentum vectors (more stable than checking entropy with theta_ff - theta_bb)
    // forward-subtree => rho_f
    // backward-subtree => rho_b
    // combined subtrees => rho
    constexpr uint8_t n_rho_cached = 3;
    arma::mat::fixed<n_params, n_rho_cached> rho_mat(arma::fill::zeros);
    auto rho_f = rho_mat.col(0);
    auto rho_b = rho_mat.col(1);
    auto rho = rho_mat.col(2);

    // AD variables assoicated with forward/backwardmost and current position.
    // - backwardmost (bb) and forwardmost (ff) are used during build_tree
    // - current is used for adaptation
    std::vector<ad::Var<double>> theta_bb_ad(n_params);
    std::vector<ad::Var<double>> theta_ff_ad(n_params);
    std::vector<ad::Var<double>> theta_curr_ad(n_params);
    mcmc::ad_bind_storage(theta_bb_ad, theta_bb, theta_bb_adj);
    mcmc::ad_bind_storage(theta_ff_ad, theta_ff, theta_ff_adj);
    mcmc::ad_bind_storage(theta_curr_ad, theta_curr, theta_curr_adj);

    // keys needed to construct a correct AD expression from model
    // key: address of original variable tags
    std::vector<const void*> keys;
    mcmc::get_keys(model, keys);

    // AD Expressions for L(theta) (log-pdf up to constant at theta)
    // Note that these expressions are the only ones used ever.
    auto theta_bb_ad_expr = model.ad_log_pdf(keys, theta_bb_ad);
    auto theta_ff_ad_expr = model.ad_log_pdf(keys, theta_ff_ad);
    auto theta_curr_ad_expr = model.ad_log_pdf(keys, theta_curr_ad);
    
    // initializes first sample into theta_curr
    // TODO: allow users to choose how to initialize first point?
    mcmc::init_sample(model, theta_curr, gen);

    // initialize current potential (will be "previous" starting in for-loop)
    double potential_prev = -ad::evaluate(theta_curr_ad_expr);

    // initialize momentum handler
    using var_adapter_policy_t = typename 
        nuts_config_traits<NUTSConfigType>::var_adapter_policy_t;
    mcmc::MomentumHandler<var_adapter_policy_t> momentum_handler(n_params);

    // initialize step adapter
    const double log_eps = std::log(
        mcmc::find_reasonable_epsilon<n_params>(
            1., // initial epsilon
            theta_curr_ad_expr, theta_curr, 
            theta_curr_adj, momentum_handler)); 
    mcmc::StepAdapter step_adapter(log_eps);        // initialize step adapter with initial log-epsilon
    step_adapter.step_config = config.step_config;  // copy step configs from user

    // initialize variance adapter
    mcmc::VarAdapter<var_adapter_policy_t> var_adapter(
            n_params, config.warmup, config.var_config.init_buffer,
            config.var_config.term_buffer, config.var_config.window_base
            );

    auto logger = util::ProgressLogger(config.n_samples + config.warmup, "NUTS");

    for (size_t i = 0; i < config.n_samples + config.warmup; ++i) {
        logger.printProgress(i);

        // re-initialize vectors to current theta as the "root" of tree
        theta_bb = theta_curr;
        theta_ff = theta_bb;
        mcmc::reset_autodiff(theta_bb_ad_expr, theta_bb_adj); 
        theta_ff_adj = theta_bb_adj;   // no need to differentiate again

        // initialize values for multinomial sampling
        // this is the total log sum weight over full tree
        double log_sum_weight = 0.;

        // initialize values used to adapt stepsize
        size_t n_leapfrog = 0;
        double sum_metro_prob = 0.;

        // p ~ N(0, M) (depending on momentum handler)
        momentum_handler.sample(p_bb); 
        p_bf = p_bb;
        p_fb = p_bb;
        p_ff = p_bb;

        // scaled p by hamiltonian dkinetic_dr
        p_bb_scaled = momentum_handler.dkinetic_dr(p_bb);
        p_bf_scaled = p_bb_scaled;
        p_fb_scaled = p_bb_scaled;
        p_ff_scaled = p_bb_scaled;

        // re-initialize integrated momentum vectors
        rho = p_bb;

        const double kinetic = momentum_handler.kinetic(p_bb);
        const double ham_prev = mcmc::hamiltonian(potential_prev, kinetic);

        // Note that this object can be reused since all members
        // are guaranteed to overwritten by build_tree.
        mcmc::TreeOutput output;

        for (size_t depth = 0; depth < config.max_depth; ++depth) {

            // TODO: optimization with rho's and copying
            // zero-out subtree integrated momentum vectors
            rho_b.zeros();
            rho_f.zeros();

            double log_sum_weight_subtree = std::numeric_limits<double>::lowest();
            
            int8_t v = 2 * direction_sampler(gen) - 1; // -1 or 1
            if (v == -1) {
                auto input = mcmc::TreeInput(
                    // position information to update
                    theta_bb_ad_expr, theta_bb, theta_bb_adj, 
                    theta_prime, p_bb,
                    // momentum vectors to update
                    p_bf, p_bb, p_bf_scaled, p_bb_scaled, rho_b,
                    // stats to update to adapt step size at the end
                    n_leapfrog, log_sum_weight_subtree, sum_metro_prob, 
                    // other miscellaneous variables
                    v, std::exp(step_adapter.log_eps), ham_prev
                );
                rho_f = rho;
                // TODO: optimization to avoid these copies
                p_fb = p_bb;
                p_fb_scaled = p_bb_scaled;

                output = mcmc::build_tree<n_params>(input, depth, 
                                           unif_sampler, gen, momentum_handler);
            } else {
                auto input = mcmc::TreeInput(
                    // correct position information to update
                    theta_ff_ad_expr, theta_ff, theta_ff_adj, 
                    theta_prime, p_ff,
                    // correct momentum vectors to update
                    p_fb, p_ff, p_fb_scaled, p_ff_scaled, rho_f,
                    // stats to update to adapt step size at the end
                    n_leapfrog, log_sum_weight_subtree, sum_metro_prob, 
                    // other miscellaneous variables
                    v, std::exp(step_adapter.log_eps), ham_prev
                );
                rho_b = rho;
                p_bf = p_ff;
                p_bf_scaled = p_ff_scaled;

                output = mcmc::build_tree<n_params>(input, depth, 
                                           unif_sampler, gen, momentum_handler);
            }

            // early break if starting to U-Turn
            if (!output.valid) break;
            
            // if new subtree's weight is greater than previous subtree's weight
            // always accept!
            if (log_sum_weight_subtree > log_sum_weight) {
                theta_curr = theta_prime;
                potential_prev = output.potential;
            } else {
                double p = std::exp(log_sum_weight_subtree - log_sum_weight);
                if (mcmc::accept_or_reject(p, unif_sampler, gen)) {
                    theta_curr = theta_prime;
                    potential_prev = output.potential;
                }
            }

            // update total log_sum_weight
            log_sum_weight = math::lse(log_sum_weight, log_sum_weight_subtree);

            // check if proposals are still 
            // - entroping in the full tree
            // - entroping from backwards-subtree to forwards-subtree
            // - entroping from forwards-subtree to backwards-subtree
            // This is a much stronger than the original paper's entropy condition.
            // This most likely reduces the depth to avoid unnecessary computation.
            
            rho = rho_b + rho_f;

            bool valid = 
                mcmc::check_entropy(rho, p_bb_scaled, p_ff_scaled) &&
                mcmc::check_entropy(rho_b + p_fb, p_bb_scaled, p_fb_scaled) &&
                mcmc::check_entropy(p_bf + rho_f, p_bf_scaled, p_ff_scaled)
                ;

            if (!valid) break;

        } // end tree doubling for-loop
        
        // Warmup Adapt!
        if (i < config.warmup) {

            // epsilon dual averaging
            step_adapter.adapt(sum_metro_prob / static_cast<double>(n_leapfrog));

            // adapt variance only if adapting policy is diag_var or dense_var 
            if constexpr (std::is_same_v<var_adapter_policy_t, diag_var> ||
                          std::is_same_v<var_adapter_policy_t, dense_var>) {
                const bool update = var_adapter.adapt(theta_curr, momentum_handler.get_m_inverse());
                if (update) {
                    double log_eps = std::log(mcmc::find_reasonable_epsilon<n_params>(
                                        std::exp(step_adapter.log_eps),
                                        theta_curr_ad_expr, theta_curr, 
                                        theta_curr_adj, momentum_handler)); 
                    step_adapter.reset();
                    step_adapter.init(log_eps);
                }
            }

            // if last warmup iteration
            if (i == config.warmup - 1) {
                step_adapter.log_eps = step_adapter.log_eps_bar;
            }
        }

        // store sample theta_curr only after burning
        if (i >= config.warmup) {
            mcmc::store_sample(model, theta_curr, i - config.warmup);
        }

    } // end for-loop to sample 1 point

    std::cout << std::endl;
}

} // namespace ppl
