#pragma once
#include <autoppl/mcmc/config_base.hpp>

namespace ppl {

struct MHConfig : ConfigBase
{
    double sigma = 1.0;
    double alpha = 0.25;
};

} // namespace ppl
