#pragma once
#include <autoppl/util/traits/traits.hpp>
#include <autoppl/util/packs/ptr_pack.hpp>
#include <autoppl/util/value.hpp>
#include <autoppl/mcmc/result.hpp>

namespace ppl {
namespace mcmc {

template <class ExprType
        , class ConfigType
        , class Sampler>
inline MCMCResult<> base_mcmc(const ExprType& expr,
                              const ConfigType& config,
                              Sampler f)
{
    using program_t = util::convert_to_program_t<ExprType>;
    program_t program = expr;

    auto pack = program.activate();
    size_t n_cont = std::get<0>(pack).uc_offset;
    size_t n_disc = std::get<1>(pack).uc_offset;
    MCMCResult<Eigen::RowMajor> res(config.samples, n_cont, n_disc);

    f(program, config, pack, res); // call actual sampling algorithm and populate res

    // Note: discrete cannot be constrained, so we only need to transform continuous

    // Compute total size of purely only constrained parameters.
    // Note that this is NOT the same pack.c_offset after activating.
    // The latter is always >= the former, but may be > (see PosDef constraint for example).
    size_t n_cont_c = 0;
    auto n_cont_c__ = [&](auto& eq_node) {
        auto& var = eq_node.get_variable();
        using var_t = std::decay_t<decltype(var)>;
        if constexpr (util::is_param_v<var_t> &&
                      util::var_traits<var_t>::is_cont_v) {
            n_cont_c += var.size();
        }
    };
    program.get_model().traverse(n_cont_c__);

    // Create transformed result object
    // Note that the number of cols for continuous sample is n_cont_c + 1,
    // where +1 is for the log-pdf.
    //
    // Computing logpdf solves two issues:
    // 1) we can save logpdf to get summary
    // 2) calling logpdf automatically evaluates all expressions properly
    // such that, in particular, constrained values are evaluated properly.

    MCMCResult<> t_res(config.samples, n_cont_c + 1, n_disc);
    std::swap(t_res.name, res.name);
    std::swap(t_res.warmup_time, res.warmup_time);
    std::swap(t_res.sampling_time, res.sampling_time);
    t_res.disc_samples.swap(res.disc_samples);

    // Transform every unconstrained params to constrained
    auto& cont_pack = std::get<0>(pack);
    auto& disc_pack = std::get<1>(pack);
    Eigen::Matrix<util::disc_param_t, Eigen::Dynamic, 1> disc_uc_val(disc_pack.uc_offset);
    Eigen::VectorXd cont_uc_val(cont_pack.uc_offset);
    Eigen::VectorXd cont_tp_val(cont_pack.tp_offset);
    Eigen::VectorXd cont_c_val(cont_pack.c_offset);
    Eigen::Matrix<size_t, Eigen::Dynamic, 1> cont_v_val(cont_pack.v_offset);
    cont_c_val.setZero();
    cont_v_val.setZero();

    util::cont_ptr_pack_t cont_ptr_pack;
    cont_ptr_pack.uc_val = cont_uc_val.data();
    cont_ptr_pack.tp_val = cont_tp_val.data();
    cont_ptr_pack.c_val = cont_c_val.data();
    cont_ptr_pack.v_val = cont_v_val.data();
    program.bind(cont_ptr_pack);

    util::disc_ptr_pack_t disc_ptr_pack;
    disc_ptr_pack.uc_val = disc_uc_val.data();
    program.bind(disc_ptr_pack);

    for (int i = 0; i < res.cont_samples.rows(); ++i) {
        disc_uc_val = t_res.disc_samples.row(i);
        cont_uc_val = res.cont_samples.row(i);
        auto lpdf = program.log_pdf();
        size_t offset = 0;
        auto copy__ = [&](auto& eq_node) {
            auto& var = eq_node.get_variable();
            using var_t = std::decay_t<decltype(var)>;
            if constexpr (util::is_param_v<var_t> &&
                          util::var_traits<var_t>::is_cont_v) {
                Eigen::Map<Eigen::MatrixXd> mp(nullptr, 0, 0);
                if constexpr (util::is_scl_v<var_t>) {
                    util::bind(mp, &var.get(), 1, 1);
                } else {
                    util::bind(mp, var.get().data(), 1, var.size());
                }
                t_res.cont_samples.block(i, offset, 1, var.size()) = mp;
                offset += var.size();
            }        
        };
        program.get_model().traverse(copy__);
        t_res.cont_samples(i, offset) = lpdf;
    }

    return t_res;
}

} // namespace mcmc
} // namespace ppl
