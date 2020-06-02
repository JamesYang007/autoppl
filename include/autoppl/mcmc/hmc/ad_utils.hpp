#pragma once
#include <algorithm>

namespace ppl {
namespace mcmc {

/*
 * Helper function to set storage for values and adjoints for AD variables
 * in "vars" as respective elements in "values" and "adjoints".
 */
template <class ADVecType, class MatType>
void ad_bind_storage(ADVecType& vars, 
                     MatType& values, 
                     MatType& adjoints)
{
    auto values_it = values.begin();
    auto adjoints_it = adjoints.begin();
    std::for_each(vars.begin(), vars.end(), 
            [&](auto& var) {
                var.set_value_ptr(&(*values_it));
                var.set_adjoint_ptr(&(*adjoints_it));
                ++values_it;
                ++adjoints_it;
            });
}

} // namespace mcmc
} // namespace ppl
