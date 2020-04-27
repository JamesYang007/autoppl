#pragma once
#include <cmath>
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
template <class ADVecType
        , class ADExprType
        , class SubviewType
        >
struct TreeInput
{
    using ad_vec_t = ADVecType;
    using ad_expr_t = ADExprType;
    using subview_t = SubviewType;

    using ad_vec_ref_t = std::reference_wrapper<ad_vec_t>;
    using ad_expr_ref_t = std::reference_wrapper<ad_expr_t>;
    using subview_ref_t = std::reference_wrapper<subview_t>;

    TreeInput(ad_expr_t& ad_expr,
              ad_vec_t& theta,
              subview_t& rho,
              double log_u,
              int8_t v,
              size_t j,
              double eps_prev,
              double ham_prev
               )
        : ad_expr_ref{ad_expr} 
        , theta_ref{theta}
        , rho_ref{rho}
        , log_u{log_u}
        , v{v}
        , j{j}
        , eps_prev{eps_prev}
        , ham_prev{ham_prev}
    {}

    ad_expr_ref_t ad_expr_ref;
    ad_vec_ref_t theta_ref;
    subview_ref_t rho_ref;
    double log_u;
    int8_t v;
    size_t j;
    double eps_prev;
    double ham_prev;
    bool theta_adjoint_exists = false;
};

/* 
 * Struct to pack output from calling build_tree.
 * If the optional references are not set, then build_tree
 * won't overwrite the content. Otherwise, overwrites.
 */

template <class ADVecType
        , class SubviewType
        >
struct TreeOutput
{
    template <class T>
    using opt_ref_t = std::optional<
        std::reference_wrapper<T>>;

    opt_ref_t<ADVecType> theta_minus_ref;
    opt_ref_t<SubviewType> rho_minus_ref;
    opt_ref_t<ADVecType> theta_plus_ref;
    opt_ref_t<SubviewType> rho_plus_ref;
    opt_ref_t<SubviewType> theta_prime_ref;
    size_t n_prime = 0;
    bool s_prime = false;
    double alpha = 0.;
    size_t n_alpha = 0;
    double potential_curr = 0.;
};

/*
 * Copy only values from src into dest AD variables.
 */
template <class T, class MatType>
void ad_copy_values(std::vector<ad::Var<T>>& dest,
                    const MatType& src)
{
    assert(dest.size() == arma::size(src)[0]);
    auto dest_it = dest.begin();
    std::for_each(src.begin(), src.end(), 
            [=](T src_x) mutable {
                dest_it->set_value(src_x);
                ++dest_it;
            });
}

/*
 * Copy only values from src AD variables into dest values.
 */
template <class MatType, class T>
void ad_copy_values(MatType& dest,
                    const std::vector<ad::Var<T>>& src)
{
    assert(arma::size(dest)[0] == src.size());
    std::transform(src.begin(), src.end(), dest.begin(),
            [](const ad::Var<T>& src_x) {
                return src_x.get_value();
            });
}

/*
 * Copy only values from src AD variables into dest AD variables.
 */
template <class T>
void ad_copy_values(std::vector<ad::Var<T>>& dest,
                    const std::vector<ad::Var<T>>& src)
{
    assert(dest.size() == src.size());
    auto dest_it = dest.begin();
    std::for_each(src.begin(), src.end(), 
            [=](const ad::Var<T>& src_x) mutable {
                dest_it->set_value(src_x.get_value());
                ++dest_it;
            });
}

/*
 * Checks if (theta_plus - theta_minus) dot r_minus
 * and similarly dotted with r_plus are both >= 0.
 */
template <class T, class MatType>
bool check_slice(const std::vector<ad::Var<T>>& theta_plus,
                 const std::vector<ad::Var<T>>& theta_minus,
                 const MatType& rho_plus,
                 const MatType& rho_minus)
{
    double dot_with_rho_plus = 0.;
    double dot_with_rho_minus = 0.;
    auto plus_it = theta_plus.begin();
    auto minus_it = theta_minus.begin();
    std::for_each(rho_plus.begin(), rho_plus.end(),
            [=, &dot_with_rho_plus](T elt) mutable { 
                dot_with_rho_plus += (plus_it->get_value() - minus_it->get_value()) * elt; 
                ++plus_it; ++minus_it;
            });
    std::for_each(rho_minus.begin(), rho_minus.end(),
            [=, &dot_with_rho_minus](T elt) mutable { 
                dot_with_rho_minus += (plus_it->get_value() - minus_it->get_value()) * elt; 
                ++plus_it; ++minus_it;
            });
    return (dot_with_rho_plus >= 0) &&
           (dot_with_rho_minus >= 0);
}

/*
 * Swap element-wise for vector of AD variables
 * and a matrix type.
 */
template <class T>
void swap(std::vector<ad::Var<T>>& v1,
          std::vector<ad::Var<T>>& v2)
{
    auto v2_it = v2.begin();
    std::for_each(v1.begin(), v1.end(), [=](ad::Var<T>& v1_elt) mutable {
                std::swap(v1_elt.get_value(), v2_it->get_value());
                ++v2_it;
            });
}

/*
 * Leapfrog integrator.
 * Returns pair of new hamiltonian and new potential.
 * Modifies adjoints of AD variables stored in vector that theta points.
 * Does not modify values of AD variables stored in vector that theta points.
 * Both modified for vector that out_theta points.
 * rho is not modified at all.
 * out_rho is updated with the "leaped" values.
 */
template <class ADThetaExprRefType
        , class ThetaRefType
        , class RhoRefType>
std::pair<double, double> leapfrog(ADThetaExprRefType ad_expr,
                                   ThetaRefType theta,
                                   bool& theta_adjoint_exists,
                                   const RhoRefType rho,
                                   double step,
                                   ThetaRefType out_theta,
                                   RhoRefType out_rho)
{
    double half_step = step / 2.;

    // Optimization: differentiate the first time if adjoint does not exist.
    // After this leapfrog, theta is updated with leaped values and adjoints remain.
    // They can be used in the recursion without recomputing.
    if (!theta_adjoint_exists) {
        ad::autodiff(ad_expr.get());
        theta_adjoint_exists = true;
    }

    // update out_rho
    auto rho_it = rho.get().begin();
    auto theta_it = theta.get().begin();
    out_rho.get().for_each([=](auto& out_rho_elt) mutable {
                out_rho_elt = *rho_it + half_step * theta_it->get_adjoint();
                ++rho_it;
                ++theta_it;
            });

    // swap input theta into out_theta and
    // updated out_theta into input theta
    // optimization to not build ad_expr all over again.
    auto out_rho_it = out_rho.get().begin();
    std::for_each(out_theta.get().begin(), out_theta.get().end(),
            [=](auto& out_theta_elt) mutable {
                // move old into out_theta temporarily
                out_theta_elt.set_value(theta_it->get_value()); 
                // update original theta values with new theta values
                theta_it->get_value() += step * (*out_rho_it);
                // IMPORTANT: renew adjoint
                theta_it->reset_adjoint();
                ++theta_it;
                ++out_rho_it;
            });

    // differentiate with theta_tilde
    double potential_new = ad::autodiff(ad_expr.get());
    out_rho.get().for_each([=](auto& out_rho_elt) mutable {
                out_rho_elt += half_step * theta_it->get_adjoint();
                ++theta_it;
            });

    // swap back original and the final theta-tilde
    alg::swap(theta.get(), out_theta.get());

    // get new hamiltonian
    double ham_new = potential_new - 
        0.5 * arma::dot(out_rho.get(), out_rho.get());

    return {ham_new, potential_new};
}

template <class ModelType, class InputType, class OutputType>
void build_tree(const ModelType& model, InputType& input, OutputType& output)
{
    using ad_vec_t = typename std::decay_t<InputType>::ad_vec_t;
    using subview_t = typename std::decay_t<InputType>::subview_t;
    constexpr double delta_max = 1000;  // suggested by Gelman

    const size_t n_params = input.theta_ref.get().size();

    // Base case: one leapfrog eps step in direction v
    if (input.j == 0) {
        double ham_curr = 0.; // new log hamiltonian
        double potential_curr = 0.; // new potential

        // TODO: all has_value checks can be optimized to just one

        // At least one pair of theta and rhos must be set.
        // In the end, all thetas and rhos will be set to the same values.
        if (output.theta_minus_ref.has_value() && 
            output.rho_minus_ref.has_value()) {
            std::tie(ham_curr, potential_curr) = leapfrog(
                    input.ad_expr_ref, input.theta_ref,
                    input.theta_adjoint_exists, input.rho_ref, 
                    input.v * input.eps_prev, output.theta_minus_ref.value(), 
                    output.rho_minus_ref.value());
        }
        else {
            std::tie(ham_curr, potential_curr) = leapfrog(
                    input.ad_expr_ref, input.theta_ref,
                    input.theta_adjoint_exists, input.rho_ref, 
                    input.v * input.eps_prev, output.theta_plus_ref.value(), 
                    output.rho_plus_ref.value());
        }

        output.n_prime = (input.log_u <= ham_curr);
        output.s_prime = (input.log_u < delta_max + ham_curr); 
        output.alpha = std::min(std::exp(ham_curr - input.ham_prev), 1.);
        output.n_alpha = 1;
        output.potential_curr = potential_curr;

        if (output.theta_minus_ref.has_value() && 
            output.rho_minus_ref.has_value()) {
            if (output.theta_plus_ref.has_value() &&
                output.rho_plus_ref.has_value()) {
                ad_copy_values(output.theta_plus_ref->get(),
                               output.theta_minus_ref->get());
                output.rho_plus_ref->get() =
                    output.rho_minus_ref->get();
            }
            ad_copy_values(output.theta_prime_ref->get(),
                           output.theta_minus_ref->get());
        }

        else {
            if (output.theta_minus_ref.has_value() &&
                output.rho_minus_ref.has_value()) {
                ad_copy_values(output.theta_minus_ref->get(),
                               output.theta_plus_ref->get());
                output.rho_minus_ref->get() =
                    output.rho_plus_ref->get();
            }
            ad_copy_values(output.theta_prime_ref->get(),
                           output.theta_plus_ref->get());
        }

        return;
    }

    // Recurse but we need temporary theta_tmp and rho_tmp information
    size_t current_j = --input.j;

    ad_vec_t theta_tmp;
    arma::mat rho_tmp_mat(0,1); 

    OutputType sub_output = output;
    
    const bool backward_need_tmp = (input.v == -1 && 
        !output.theta_plus_ref.has_value() &&
        !output.rho_plus_ref.has_value());

    const bool forward_need_tmp = (input.v == 1 && 
        !output.theta_minus_ref.has_value() &&
        !output.rho_minus_ref.has_value());

    // temporary storages will be used
    if (backward_need_tmp || forward_need_tmp) {
        theta_tmp.resize(n_params); 
        rho_tmp_mat.resize(n_params, 1);
    }

    subview_t rho_tmp = rho_tmp_mat.col(0);

    // if backwards direction and theta/rho plus not set,
    // set them to be the temporary containers
    if (backward_need_tmp) {
        sub_output.theta_plus_ref = theta_tmp;
        sub_output.rho_plus_ref = rho_tmp;

    } else if (forward_need_tmp) {
        sub_output.theta_minus_ref = theta_tmp;
        sub_output.rho_minus_ref = rho_tmp;
    }

    build_tree(model, input, sub_output);

    // restore current j 
    input.j = current_j;

    if (sub_output.s_prime == 1) {
        arma::mat theta_mat(n_params, 1);
        auto theta_double_prime = theta_mat.col(0);
        OutputType sub_sub_output = output;

        // no matter what, if backwards, disable plus and vice versa
        if (input.v == -1) {
            sub_sub_output.theta_plus_ref.reset();
            sub_sub_output.rho_plus_ref.reset();
        } else {
            sub_sub_output.theta_minus_ref.reset();
            sub_sub_output.rho_minus_ref.reset();
        }
        // bind theta prime reference to the newly created double prime
        sub_sub_output.theta_prime_ref = theta_double_prime;

        build_tree(model, input, sub_sub_output);

        std::uniform_real_distribution metrop_sampler(0., 1.);        
        std::mt19937 gen;
        double accept_prob = static_cast<double>(sub_sub_output.n_prime) / 
            (sub_sub_output.n_prime + sub_output.n_prime);
        if (metrop_sampler(gen) <= accept_prob) {
            output.theta_prime_ref->get() = 
                sub_sub_output.theta_prime_ref->get();
        }

        // update actual values in the original output
        output.alpha = sub_output.alpha + sub_sub_output.alpha;
        output.n_alpha = sub_output.n_alpha + sub_sub_output.n_alpha;
        output.n_prime = sub_output.n_prime + sub_sub_output.n_prime;
        output.s_prime = sub_sub_output.s_prime && check_slice(
                    sub_output.theta_plus_ref->get(),
                    sub_output.theta_minus_ref->get(),
                    sub_output.rho_plus_ref->get(),
                    sub_output.rho_minus_ref->get()
                );
    } else {
        // Otherwise, only need to copy over primitives
        output.alpha = sub_output.alpha;
        output.n_alpha = sub_output.n_alpha;
        output.n_prime = sub_output.n_prime;
        output.s_prime = sub_output.s_prime;
    } // end if
}

template <class ModelType, class MatType>
double find_reasonable_log_epsilon(ModelType& model,
                                   std::vector<const void*> keys,
                                   const MatType& theta0,
                                   size_t max_iter)
{
    double eps = 1.;
    const double diff_bound = -std::log(2);
    const size_t n_params = arma::size(theta0)[0];

    std::vector<ad::Var<double>> theta0_ad(n_params);
    std::vector<ad::Var<double>> theta_prime(n_params); // should just be double 
    ad_copy_values(theta0_ad, theta0);

    arma::mat r_mat(n_params, 2);
    auto r = r_mat.col(0);
    auto r_prime = r_mat.col(1);
    r.randn();

    auto ad_expr = model.ad_log_pdf(keys, theta0_ad);

    double ham_prev = ad::autodiff(ad_expr); // differentiate first

    // save original adjoint
    std::vector<double> orig_adj(n_params);
    std::transform(theta0_ad.begin(), theta0_ad.end(), orig_adj.begin(), 
            [](const ad::Var<double>& x) {
                return x.get_adjoint();
            });

    bool adjoint_exists = true;
    
    auto [ham_curr, potential_curr] = leapfrog(
            std::ref(ad_expr),
            std::ref(theta0_ad),
            adjoint_exists,
            std::ref(r),
            eps,
            std::ref(theta_prime),
            std::ref(r_prime));

    int a = 2*(ham_curr - ham_prev > diff_bound) - 1;

    while ((a * (ham_curr - ham_prev) > a * diff_bound) &&
            max_iter--) {
        eps *= std::pow(2, a);
        // copy back original value and adjoint
        auto orig_adj_it = orig_adj.begin();
        auto theta0_it = theta0.begin();
        std::for_each(theta0_ad.begin(), theta0_ad.end(), 
                [=](ad::Var<double>& x) mutable {
                    x.set_value(*theta0_it);
                    x.set_adjoint(*orig_adj_it);
                    ++theta0_it;
                    ++orig_adj_it;
                });

        auto pair = leapfrog(
                std::ref(ad_expr),
                std::ref(theta0_ad),
                adjoint_exists,
                std::ref(r),
                eps,
                std::ref(theta_prime),
                std::ref(r_prime));
        ham_curr = std::get<0>(pair); 
        a = 2*(ham_curr - ham_prev > diff_bound) - 1;
    }

    return std::log(eps);
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
          double delta,
          size_t n_samples,
          size_t n_adapt,
          size_t max_depth = 10,
          size_t max_init_iter = 5,
          size_t seed = 0)
{

    // initialization of meta-variables
    double log_eps_bar = 0.;
    double H_bar = 0;
    constexpr double gamma = 0.05;
    constexpr double t0 = 10;
    constexpr double kappa = 0.75;
    const size_t n_params = alg::get_n_params(model);
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
        using state_t = typename util::var_traits<var_t>::state_t;
        if (var.get_state() == state_t::parameter) {
            *keys_it = &var;
            ++keys_it;
        }
    };
    model.traverse(get_keys);
    
    // momentum matrix
    constexpr uint8_t n_rhos_cached = 3;
    arma::mat rho_mat(n_params, n_rhos_cached);
    auto rho0 = rho_mat.col(0);
    auto rho_minus = rho_mat.col(1);
    auto rho_plus = rho_mat.col(2);
    using rho_t = std::decay_t<decltype(rho0)>;

    // position theta^+ and theta^-
    // needs AD variables for gradient computation
    // TODO: make cache efficient
    using theta_ad_t = std::vector<ad::Var<double>>;
    theta_ad_t theta_minus(n_params);
    theta_ad_t theta_plus(n_params);

    // AD Expressions for theta_minus and theta_plus.
    // Note that these expressions are the only ones used ever.
    // L(theta) (potential of input theta)
    auto theta_minus_ad = model.ad_log_pdf(keys, theta_minus);
    auto theta_plus_ad = model.ad_log_pdf(keys, theta_plus);
    
    // position matrix for theta' and current theta
    constexpr uint8_t n_thetas_cached = 2;
    arma::mat theta_mat(n_params, n_thetas_cached);
    auto theta_prime = theta_mat.col(0);
    auto theta_curr = theta_mat.col(1);
    
    // initialize model tags using model specs
    // copies the initialized values into theta_curr
    // initialize potential energy
    auto theta_curr_it = theta_curr.begin();
    alg::init_params(model, gen);    
    double potential_prev = 0.;
    auto copy_params_potential = [=, &potential_prev](const auto& eq_node) mutable {
        const auto& var = eq_node.get_variable();
        const auto& dist = eq_node.get_distribution();
        using var_t = std::decay_t<decltype(var)>;
        using state_t = typename util::var_traits<var_t>::state_t;
        if (var.get_state() == state_t::parameter) {
            *theta_curr_it = var.get_value(); 
            ++theta_curr_it;
            potential_prev += dist.log_pdf_no_constant(var.get_value());
        }
    };
    model.traverse(copy_params_potential);

    // initialize rest of the metavariables
    double log_eps = alg::find_reasonable_log_epsilon(
            model, keys, theta_curr, max_init_iter);  
    const double mu = std::log(10) + log_eps;

    // tree output struct type
    using tree_output_t = alg::TreeOutput<theta_ad_t, rho_t>;

    for (size_t i = 0; i < n_samples; ++i) {

        // rho0 ~ N(0, I)
        rho_mat.randn(); 

        // u ~ Uniform[0, exp(L(theta^{m-1}) - 0.5 * r0^2)] 
        const double ham_prev = potential_prev - 0.5 * arma::dot(rho0, rho0);
        std::uniform_real_distribution unif_sampler(0., std::exp(ham_prev));
        double log_u = std::log(unif_sampler(gen));

        alg::ad_copy_values(theta_minus, theta_curr);
        alg::ad_copy_values(theta_plus, theta_curr);
        rho_minus = rho0;
        rho_plus = rho0;
        size_t j = 0;
        size_t n = 1;
        bool s = true;

        tree_output_t output;
        output.theta_prime_ref = theta_prime;

        bool update_potential = false; // if new candidate accepted, potential should be updated

        while (s && (j < max_depth)) {
            int8_t v = 2 * direction_sampler(gen) - 1; // -1 or 1
            if (v == -1) {
                auto input = alg::TreeInput(
                    theta_minus_ad, theta_minus, rho_minus,
                    log_u, v, j, std::exp(log_eps), ham_prev
                );
                output.theta_minus_ref = theta_minus;
                output.rho_minus_ref = rho_minus;
                output.theta_plus_ref.reset();
                output.rho_plus_ref.reset();
                alg::build_tree(model, input, output);
            } else {
                auto input = alg::TreeInput(
                    theta_plus_ad, theta_plus, rho_plus,
                    log_u, v, j, std::exp(log_eps), ham_prev
                );
                output.theta_minus_ref.reset();
                output.rho_minus_ref.reset();
                output.theta_plus_ref = theta_plus;
                output.rho_plus_ref = rho_plus;
                alg::build_tree(model, input, output);
            }

            if (output.s_prime) {
                if (metrop_sampler(gen) <= output.n_prime/static_cast<double>(n)) {
                    theta_curr = theta_prime;
                    update_potential = true;
                }
            }

            n += output.n_prime;
            s = output.s_prime && alg::check_slice(
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

        if (update_potential) potential_prev = output.potential_curr;

        // store sample theta_curr
        auto theta_curr_it = theta_curr.begin();
        auto store_sample = [=](auto& eq_node) mutable {
            auto& var = eq_node.get_variable();
            using var_t = std::decay_t<decltype(var)>;
            using state_t = typename util::var_traits<var_t>::state_t;
            if (var.get_state() == state_t::parameter) {
                auto storage_ptr = var.get_storage();
                storage_ptr[i] = *theta_curr_it;
                ++theta_curr_it;
            }
        };
        model.traverse(store_sample);
        
    } // end for
}

} // namespace ppl
