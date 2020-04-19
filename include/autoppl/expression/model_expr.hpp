#pragma once

namespace ppl {

template <class Derived>
struct ModelExpr
{
    Derived& self() 
    { return static_cast<Derived&>(*this); }

    const Derived& self() const
    { return static_cast<const Derived&>(*this); }
};

} // namespace ppl
