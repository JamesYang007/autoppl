#pragma once
#include <cstddef>
#include <autoppl/mcmc/sampler_tools.hpp>

namespace ppl {

struct ConfigBase
{
    size_t warmup = 1000;
    size_t samples = 1000;
    size_t seed = mcmc::random_seed();
    bool prune = true;
};

} // namespace ppl
