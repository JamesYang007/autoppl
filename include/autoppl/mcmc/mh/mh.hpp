#pragma once
#include <random>
#include <autoppl/util/logging.hpp>
#include <autoppl/util/traits/traits.hpp>
#include <autoppl/util/time/stopwatch.hpp>
#include <autoppl/util/packs/ptr_pack.hpp>
#include <autoppl/mcmc/sampler_tools.hpp>
#include <autoppl/mcmc/result.hpp>
#include <autoppl/mcmc/mh/config.hpp>
#include <autoppl/mcmc/base_mcmc.hpp>

namespace ppl {
namespace mcmc {

/**
 * Metropolis-Hastings algorithm to sample from posterior distribution.
 * Any variables that model references which are parameters are sampled.
 *
 * @tparam  ProgramType     program expression type
 * @tparam  OffsetPackType  offset pack type (likely util::OffsetPack)
 * @param   program         program expression
 * @param   config          configuration object
 * @param   pack            offset pack from activating program expression
 * @param   res             sampling result object to populate
 */

template <class ProgramType
        , class OffsetPackType
        , class MCMCResultType>
inline void mh_(const ProgramType& program,
                const MHConfig& config,
                const OffsetPackType& pack,
                MCMCResultType& res)
{
    ProgramType program_curr = program;   // will be bound to curr
    ProgramType program_cand = program;   // will be bound to cand

    // data structure to keep track of param candidates
    using cont_vec_t = Eigen::Matrix<util::cont_param_t, Eigen::Dynamic, 1>;
    using disc_vec_t = Eigen::Matrix<util::disc_param_t, Eigen::Dynamic, 1>;
    using visit_vec_t = Eigen::Matrix<size_t, Eigen::Dynamic, 1>;

    cont_vec_t cont_curr(std::get<0>(pack).uc_offset); // total number of offsets is the number of parameters
    cont_vec_t cont_cand(std::get<0>(pack).uc_offset); 
    cont_vec_t cont_tp(std::get<0>(pack).tp_offset);
    cont_vec_t cont_constrained(std::get<0>(pack).c_offset);
    visit_vec_t cont_visit(std::get<0>(pack).v_offset);
    cont_tp.setZero();
    cont_constrained.setZero();
    cont_visit.setZero();

    disc_vec_t disc_curr(std::get<1>(pack).uc_offset); 
    disc_vec_t disc_cand(std::get<1>(pack).uc_offset); 
    disc_vec_t disc_tp(std::get<1>(pack).tp_offset);

    util::cont_ptr_pack_t cont_ptr_pack;
    cont_ptr_pack.uc_val = cont_curr.data();
    cont_ptr_pack.c_val = cont_constrained.data();
    cont_ptr_pack.v_val = cont_visit.data();
    cont_ptr_pack.tp_val = cont_tp.data();

    util::disc_ptr_pack_t disc_ptr_pack;
    disc_ptr_pack.uc_val = disc_curr.data();
    disc_ptr_pack.tp_val = disc_tp.data();

    program_curr.bind(cont_ptr_pack);
    program_curr.bind(disc_ptr_pack);

    cont_ptr_pack.uc_val = cont_cand.data();
    disc_ptr_pack.uc_val = disc_cand.data();
    program_cand.bind(cont_ptr_pack);
    program_cand.bind(disc_ptr_pack);

    std::uniform_real_distribution metrop_sampler(0., 1.);
    std::discrete_distribution disc_sampler({config.alpha, 1-2*config.alpha, config.alpha});
    std::normal_distribution norm_sampler(0., config.sigma);
    std::mt19937 gen(config.seed);

    // references avoid making copies when swapping at the end of for-loop
    std::reference_wrapper<ProgramType> program_curr_ref(program_curr);
    std::reference_wrapper<ProgramType> program_cand_ref(program_cand);

    program_curr_ref.get().init_params(gen, config.prune);
    double curr_log_pdf = program_curr.log_pdf();

    // construct miscellaneous objects 
    auto logger = util::ProgressLogger(config.samples + config.warmup, "Metropolis-Hastings");
    util::StopWatch<> stopwatch_warmup;
    util::StopWatch<> stopwatch_sampling;

    // start timing warmup
    stopwatch_warmup.start();

    for (size_t iter = 0; iter < config.samples + config.warmup; ++iter) {

        // if warmup is finished, stop timing warmup and start timing sampling
        if (iter == config.warmup) {
            stopwatch_warmup.stop();
            stopwatch_sampling.start();
        }

        logger.printProgress(iter);

        double log_alpha = -curr_log_pdf;

        // generate next candidates
        cont_cand = cont_curr + cont_vec_t::NullaryExpr(cont_cand.size(), 
                [&]() { return norm_sampler(gen); });
        disc_cand = disc_curr + disc_vec_t::NullaryExpr(disc_cand.size(),
                [&]() { return disc_sampler(gen) - 1; });

        // compute next candidate log pdf and update log_alpha
        double cand_log_pdf = program_cand_ref.get().log_pdf();
        log_alpha += cand_log_pdf;
        bool accept = (std::log(metrop_sampler(gen)) <= log_alpha);

        if (accept) {
            cont_curr.swap(cont_cand);
            disc_curr.swap(disc_cand);
            std::swap(program_curr_ref, program_cand_ref);
            curr_log_pdf = cand_log_pdf;
        }

        if (iter >= config.warmup) {
            res.cont_samples.row(iter-config.warmup) = cont_curr;
            res.disc_samples.row(iter-config.warmup) = disc_curr;
        }
    }

    // stop timing sampling
    stopwatch_sampling.stop();

    // save output results
    res.warmup_time = stopwatch_warmup.elapsed();
    res.sampling_time = stopwatch_sampling.elapsed();
}

} // namespace mcmc

template <class ExprType>
inline auto mh(const ExprType& expr,
               const MHConfig& config = MHConfig())
{
    return mcmc::base_mcmc(expr, config, 
            [](const auto& program, const auto& config,
               const auto& pack, auto& res) {
                res.name = "mh";
                mcmc::mh_(program, config, pack, res);
            });
}

} // namespace ppl
