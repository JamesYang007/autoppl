#pragma once

namespace ppl {
namespace mcmc{
    
/**
 * Compute Hamiltonian given potential and kinetic energy.
 * Returns simply the sum of the two.
 */
inline double hamiltonian(double potential, double kinetic)
{ return potential + kinetic; }

} // namespace mcmc
} // namespace ppl
