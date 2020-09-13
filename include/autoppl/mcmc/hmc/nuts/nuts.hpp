#pragma once
#include <type_traits>
#include <Eigen/Dense>
#include <fastad_bits/reverse/core/var_view.hpp>
#include <fastad_bits/reverse/core/eval.hpp>
#include <autoppl/util/traits/var_traits.hpp>
#include <autoppl/util/packs/ptr_pack.hpp>
#include <autoppl/util/logging.hpp>
#include <autoppl/util/time/stopwatch.hpp>
#include <autoppl/math/math.hpp>
#include <autoppl/mcmc/sampler_tools.hpp>
#include <autoppl/mcmc/result.hpp>
#include <autoppl/mcmc/base_mcmc.hpp>
#include <autoppl/mcmc/hmc/nuts/tree_utils.hpp>
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
    return rho.dot(p_beg_scaled) > 0 &&
           rho.dot(p_end_scaled) > 0;
}

/**
 * Building binary tree for sampling candidates.
 * Helper function to obtain the forward/backward-most position and momentum.
 * Accept/reject policy is based on UniformDistType parameter and GenType
 *
 * Note that the caller, i.e. nuts(), MUST have theta_adj already pre-computed
 * (theta_adj is a member of input and input will be an instance of TreeInput).
 *
 * @param   n_params            number of (continuous) parameters
 * @param   input               TreeInput-like input object
 * @param   depth               current depth of building tree
 * @param   unif_sampler        an object like std::uniform_distribution(0,1)
 *                              used for metropolis acceptance
 * @param   gen                 rng device
 * @param   momentum_handler    MomentumHandler-like object to compute 
 *                              kinetic energy and momentum
 * @param   tree_cache          pointer to cache memory that will be used by build_tree.
 *                              The array of doubles must be of size n_params * 7 * max_depth.
 */
template <class InputType
        , class UniformDistType
        , class GenType
        , class MomentumHandlerType
    >
TreeOutput build_tree(size_t n_params, 
                      InputType& input, 
                      uint8_t depth,
                      UniformDistType& unif_sampler,
                      GenType& gen,
                      MomentumHandlerType& momentum_handler,
                      double* tree_cache)
{
    constexpr double delta_max = 1000;  // suggested by Gelman

    // base case
    if (depth == 0) {
        double new_potential = leapfrog(input.ad_expr_ref.get(),
                                        input.theta_ref.get(),
                                        input.theta_adj_ref.get(),
                                        input.tp_adj_ref.get(),
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
        if (std::isnan(new_ham)) { new_ham = math::inf<double>; }
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
    Eigen::Map<Eigen::VectorXd> p_end_inner(tree_cache, n_params);
    Eigen::Map<Eigen::VectorXd> p_end_scaled_inner(tree_cache + n_params, n_params);
    Eigen::Map<Eigen::VectorXd> rho_first(tree_cache + 2*n_params, n_params);
    rho_first.setZero();
    double log_sum_weight_first = math::neg_inf<double>;

    tree_cache += 3 * n_params; // update position of tree cache

    // create a new input for first recursion
    // some references have to rebound
    InputType first_input = input; 
    first_input.p_end_ref = p_end_inner;
    first_input.p_end_scaled_ref = p_end_scaled_inner;
    first_input.rho_ref = rho_first;
    first_input.log_sum_weight_ref = log_sum_weight_first;

    // build first subtree
    TreeOutput first_output = 
        build_tree(n_params, first_input, depth - 1, 
                   unif_sampler, gen, momentum_handler,
                   tree_cache);

    // if first subtree is already invalid, early exit
    // note that caller will break out of doubling process now,
    // so we do not have to worry about updating the other momentum vectors
    if (!first_output.valid) { return first_output; }

    // second recursion
    Eigen::Map<Eigen::VectorXd> theta_double_prime(tree_cache, n_params);
    Eigen::Map<Eigen::VectorXd> p_beg_inner(tree_cache + n_params, n_params);
    Eigen::Map<Eigen::VectorXd> p_beg_scaled_inner(tree_cache + 2*n_params, n_params);
    Eigen::Map<Eigen::VectorXd> rho_second(tree_cache + 3*n_params, n_params);
    rho_second.setZero();
    double log_sum_weight_second = math::neg_inf<double>;

    tree_cache += 4 * n_params;

    // create a new input for second recursion
    InputType second_input = input;
    second_input.theta_prime_ref = theta_double_prime;
    second_input.p_beg_ref = p_beg_inner;
    second_input.p_beg_scaled_ref = p_beg_scaled_inner;
    second_input.rho_ref = rho_second;
    second_input.log_sum_weight_ref = log_sum_weight_second;

    // build second subtree
    TreeOutput second_output = 
        build_tree(n_params, second_input, depth - 1, 
                   unif_sampler, gen, momentum_handler,
                   tree_cache);

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
 *
 * @param   eps                 initial epsilon (see Gelman's paper)
 * @param   ad_expr             AD expression bound to theta and theta_adj
 * @param   theta               vector of theta values
 * @param   theta_adj           vector of theta adjoints
 * @param   gen                 rng device
 * @param   momentum_handler    MomentumHandler-like object 
 */
template <class ADExprType
        , class MatType
        , class GenType
        , class MomentumHandlerType>
double find_reasonable_epsilon(double eps,
                               ADExprType& ad_expr,
                               MatType& theta,
                               MatType& theta_adj,
                               MatType& tp_adj,
                               GenType& gen,
                               MomentumHandlerType& momentum_handler)
{
    // See (STAN) for reference: if epsilon is way out of bounds, just return eps
    if (eps <= 0 || eps > 1e7) return eps;

    const double diff_bound = std::log(0.8);

    size_t n_params = theta.rows(); // theta is expected to be vector-like

    Eigen::MatrixXd mat(n_params, 3);
    Eigen::Map<Eigen::VectorXd> r(mat.col(0).data(), n_params);
    Eigen::Map<Eigen::VectorXd> theta_orig(mat.col(1).data(), n_params);
    Eigen::Map<Eigen::VectorXd> theta_adj_orig(mat.col(2).data(), n_params);

    // sample momentum vector based on handler
    momentum_handler.sample(r, gen);

    // differentiate first to get adjoints and hamiltonian
    const double potential_orig = -ad::autodiff(ad_expr); 
    double kinetic_orig = momentum_handler.kinetic(r);
    double ham_orig = hamiltonian(potential_orig, kinetic_orig);

    // save original value and adjoint
    theta_orig = theta;
    theta_adj_orig = theta_adj;
    
    // get current hamiltonian after leapfrog
    double potential_curr = leapfrog(
            ad_expr, theta, theta_adj, tp_adj,
            r, momentum_handler, eps, true);
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
        momentum_handler.sample(r, gen);
        kinetic_orig = momentum_handler.kinetic(r);
        ham_orig = hamiltonian(potential_orig, kinetic_orig);

        // leapfrog and compute current hamiltonian
        potential_curr = leapfrog(
                ad_expr, theta, theta_adj, tp_adj,
                r, momentum_handler, eps, true);
        kinetic_curr = momentum_handler.kinetic(r);
        ham_curr = hamiltonian(potential_curr, kinetic_curr);

    }

    // copy back original value and adjoint
    theta = theta_orig;
    theta_adj = theta_adj_orig;

    return eps;
}

/**
 * No-U-Turn Sampler (NUTS)
 *
 * User must ensure that the program does not have any discrete parameters.
 * Discrete data is allowed.
 *
 * @param   program     program expression used to determine log-pdf
 * @param   config      NUTS configuration object
 * @param   pack        offset pack result of activating program.
 *                      It will likely be util::OffsetPack where each offset
 *                      value is equivalent to the total number of values needed,
 *                      i.e. if pack.uc_offset is 10, there is exactly 10 unconstrained values
 *                      for the program.
 * @param   res         result object of calling NUTS that will be populated with samples and other information.
 */

template <class ProgramType
        , class OffsetPackType
        , class MCMCResultType
        , class NUTSConfigType = NUTSConfig<>>
void nuts_(ProgramType& program, 
           const NUTSConfigType& config,
           const OffsetPackType& pack,
           MCMCResultType& res)
{
    assert(std::get<1>(pack).uc_offset == 0);
    assert(std::get<1>(pack).tp_offset == 0);
    assert(std::get<1>(pack).c_offset == 0);
    assert(std::get<1>(pack).v_offset == 0);

    auto& offset_pack = std::get<0>(pack);
    size_t n_params = offset_pack.uc_offset;

    // initialization of meta-variables
    std::mt19937 gen(config.seed);
    std::uniform_int_distribution direction_sampler(0, 1);
    std::uniform_real_distribution unif_sampler(0., 1.);

    // Transformed parameters, constrained parameter, visit count cache
    // This can be shared across all AD expressions since only one expression
    // will be evaluated at a time.
    Eigen::MatrixXd tp_mat(offset_pack.tp_offset, 2);
    Eigen::Map<Eigen::VectorXd> tp_val(tp_mat.col(0).data(), offset_pack.tp_offset);
    Eigen::Map<Eigen::VectorXd> tp_adj(tp_mat.col(1).data(), offset_pack.tp_offset);
    Eigen::VectorXd constrained(offset_pack.c_offset);
    Eigen::Matrix<size_t, Eigen::Dynamic, 1> visit(offset_pack.v_offset);
    tp_mat.setZero();
    constrained.setZero();
    visit.setZero();

    // momentum matrix (for stability reasons we require knowing 4 momentum)
    // left-subtree backwardmost momentum => bb
    // left-subtree forwardmost momentum => bf
    // right-subtree backwardmost momentum => fb
    // right-subtree forwardmost momentum => ff
    // scaled versions are based on hamiltonian adjusted covariance matrix
    Eigen::MatrixXd cache_mat(n_params, 18);
    cache_mat.setZero();
    Eigen::Map<Eigen::VectorXd> p_bb(cache_mat.col(0).data(), n_params);
    Eigen::Map<Eigen::VectorXd> p_bb_scaled(cache_mat.col(1).data(), n_params);
    Eigen::Map<Eigen::VectorXd> p_bf(cache_mat.col(2).data(), n_params);
    Eigen::Map<Eigen::VectorXd> p_bf_scaled(cache_mat.col(3).data(), n_params);
    Eigen::Map<Eigen::VectorXd> p_fb(cache_mat.col(4).data(), n_params);
    Eigen::Map<Eigen::VectorXd> p_fb_scaled(cache_mat.col(5).data(), n_params);
    Eigen::Map<Eigen::VectorXd> p_ff(cache_mat.col(6).data(), n_params);
    Eigen::Map<Eigen::VectorXd> p_ff_scaled(cache_mat.col(7).data(), n_params);

    // position matrix for thetas and adjoints
    Eigen::Map<Eigen::VectorXd> theta_bb(cache_mat.col(8).data(), n_params);
    Eigen::Map<Eigen::VectorXd> theta_bb_adj(cache_mat.col(9).data(), n_params);
    Eigen::Map<Eigen::VectorXd> theta_ff(cache_mat.col(10).data(), n_params);
    Eigen::Map<Eigen::VectorXd> theta_ff_adj(cache_mat.col(11).data(), n_params);
    Eigen::Map<Eigen::VectorXd> theta_curr(cache_mat.col(12).data(), n_params);
    Eigen::Map<Eigen::VectorXd> theta_curr_adj(cache_mat.col(13).data(), n_params);
    Eigen::Map<Eigen::VectorXd> theta_prime(cache_mat.col(14).data(), n_params);

    // integrated momentum vectors (more stable than checking entropy with theta_ff - theta_bb)
    // forward-subtree => rho_f
    // backward-subtree => rho_b
    // combined subtrees => rho
    Eigen::Map<Eigen::VectorXd> rho_f(cache_mat.col(15).data(), n_params);
    Eigen::Map<Eigen::VectorXd> rho_b(cache_mat.col(16).data(), n_params);
    Eigen::Map<Eigen::VectorXd> rho(cache_mat.col(17).data(), n_params);

    // build-tree helper function cache line
    Eigen::VectorXd tree_cache(n_params * 7 * config.max_depth);
    tree_cache.setZero();

    // AD Expressions for L(theta) (log-pdf up to constant at theta)
    // Note that these expressions are the only ones used ever.
    auto theta_bb_ad_expr = program.ad_log_pdf(util::make_ptr_pack(
            theta_bb.data(), theta_bb_adj.data(), 
            tp_val.data(), tp_adj.data(),
            constrained.data(), visit.data() ));
    auto theta_ff_ad_expr = program.ad_log_pdf(util::make_ptr_pack(
            theta_ff.data(), theta_ff_adj.data(),
            tp_val.data(), tp_adj.data(),
            constrained.data(), visit.data() ));
    auto theta_curr_ad_expr = program.ad_log_pdf(util::make_ptr_pack(
            theta_curr.data(), theta_curr_adj.data(),
            tp_val.data(), tp_adj.data(),
            constrained.data(), visit.data() ));

    // bind every AD expression to the same cache line
    auto size_pack = theta_bb_ad_expr.bind_cache_size();
    Eigen::VectorXd ad_val_buf(size_pack(0));
    Eigen::VectorXd ad_adj_buf(size_pack(1));
    theta_bb_ad_expr.bind_cache({ad_val_buf.data(), ad_adj_buf.data()});
    theta_ff_ad_expr.bind_cache({ad_val_buf.data(), ad_adj_buf.data()});
    theta_curr_ad_expr.bind_cache({ad_val_buf.data(), ad_adj_buf.data()});
    
    // initializes first sample into theta_curr
    // TODO: allow users to choose how to initialize first point?
    program.bind(util::make_ptr_pack(
                theta_curr.data(), nullptr,
                tp_val.data(), nullptr,
                constrained.data(), visit.data()));
    program.init_params(gen, config.prune);

    // initialize current potential (will be "previous" starting in for-loop)
    double potential_prev = -ad::evaluate(theta_curr_ad_expr);

    // initialize momentum handler
    using var_adapter_policy_t = typename 
        nuts_config_traits<NUTSConfigType>::var_adapter_policy_t;
    mcmc::MomentumHandler<var_adapter_policy_t> momentum_handler(n_params);

    // initialize step adapter
    const double log_eps = std::log(
        mcmc::find_reasonable_epsilon(
            1., // initial epsilon
            theta_curr_ad_expr, theta_curr, 
            theta_curr_adj, tp_adj,
            gen, momentum_handler)); 
    mcmc::StepAdapter step_adapter(log_eps);        // initialize step adapter with initial log-epsilon
    step_adapter.step_config = config.step_config;  // copy step configs from user

    // initialize variance adapter
    mcmc::VarAdapter<var_adapter_policy_t> var_adapter(
            n_params, config.warmup, config.var_config.init_buffer,
            config.var_config.term_buffer, config.var_config.window_base
            );

    // construct miscellaneous objects 
    auto logger = util::ProgressLogger(config.samples + config.warmup, "NUTS");
    util::StopWatch<> stopwatch_warmup;
    util::StopWatch<> stopwatch_sampling;

    // start timing warmup
    stopwatch_warmup.start();

    for (size_t i = 0; i < config.samples + config.warmup; ++i) {

        // if warmup is finished, stop timing warmup and start timing sampling
        if (i == config.warmup) {
            stopwatch_warmup.stop();
            stopwatch_sampling.start();
        }

        logger.printProgress(i);

        // re-initialize vectors to current theta as the "root" of tree
        theta_bb = theta_curr;
        theta_ff = theta_bb;
        mcmc::reset_autodiff(theta_bb_ad_expr, theta_bb_adj, tp_adj); 
        theta_ff_adj = theta_bb_adj;   // no need to differentiate again

        // initialize values for multinomial sampling
        // this is the total log sum weight over full tree
        double log_sum_weight = 0.;

        // initialize values used to adapt stepsize
        size_t n_leapfrog = 0;
        double sum_metro_prob = 0.;

        // p ~ N(0, M) (depending on momentum handler)
        momentum_handler.sample(p_bb, gen); 
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

            // zero-out subtree integrated momentum vectors
            rho_b.setZero();
            rho_f.setZero();

            double log_sum_weight_subtree = math::neg_inf<double>;
            
            int8_t v = 2 * direction_sampler(gen) - 1; // -1 or 1
            if (v == -1) {
                auto input = mcmc::TreeInput(
                    // position information to update
                    theta_bb_ad_expr, theta_bb, theta_bb_adj, tp_adj,
                    theta_prime, p_bb,
                    // momentum vectors to update
                    p_bf, p_bb, p_bf_scaled, p_bb_scaled, rho_b,
                    // stats to update to adapt step size at the end
                    n_leapfrog, log_sum_weight_subtree, sum_metro_prob, 
                    // other miscellaneous variables
                    v, std::exp(step_adapter.log_eps), ham_prev
                );
                rho_f = rho;
                p_fb = p_bb;
                p_fb_scaled = p_bb_scaled;

                output = mcmc::build_tree(n_params, input, depth, 
                                          unif_sampler, gen, momentum_handler,
                                          tree_cache.data());
            } else {
                auto input = mcmc::TreeInput(
                    // correct position information to update
                    theta_ff_ad_expr, theta_ff, theta_ff_adj, tp_adj,
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

                output = mcmc::build_tree(n_params, input, depth, 
                                          unif_sampler, gen, momentum_handler,
                                          tree_cache.data());
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
                    double log_eps = std::log( mcmc::find_reasonable_epsilon(
                                        std::exp(step_adapter.log_eps),
                                        theta_curr_ad_expr, theta_curr, 
                                        theta_curr_adj, tp_adj,
                                        gen, momentum_handler) ); 
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
            res.cont_samples.row(i-config.warmup) = theta_curr;
        }

    } // end for-loop to sample 1 point

    // stop timing sampling
    stopwatch_sampling.stop();

    // save output results
    res.warmup_time = stopwatch_warmup.elapsed();
    res.sampling_time = stopwatch_sampling.elapsed();
}

} // namespace mcmc

template <class ExprType
        , class NUTSConfigType = NUTSConfig<>>
inline auto nuts(const ExprType& expr, 
                 const NUTSConfigType& config = NUTSConfigType())
{
    return mcmc::base_mcmc(expr, config, 
            [](auto& program, const auto& config,
               const auto& pack, auto& res) {
                res.name = "nuts";
                mcmc::nuts_(program, config, pack, res);
            });
}

} // namespace ppl
