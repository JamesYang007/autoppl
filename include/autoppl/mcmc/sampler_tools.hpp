#pragma once
#include <chrono>

namespace ppl {
namespace mcmc {

/**
 * Get current time in milliseconds for random seeding.
 */
inline size_t random_seed()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::system_clock::now().time_since_epoch()).count();
}

/**
 * Accepts or rejects with given probability using UniformDistType
 * object that works with GenType.
 * The uniform sampler must sample from [0,1].
 */
template <class UniformDistType, class GenType>
inline bool accept_or_reject(double p, 
                             UniformDistType&& unif_sampler,
                             GenType&& gen)
{
    return (unif_sampler(gen) <= p);
}

} // namespace mcmc
} // namespace ppl
