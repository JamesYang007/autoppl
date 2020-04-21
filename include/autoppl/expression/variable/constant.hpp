#pragma once

namespace ppl {
namespace expr {

template <class ValueType>
struct Constant
{
    using value_t = ValueType;
    Constant(value_t c)
        : c_{c}
    {}
    operator value_t() const { return c_; }

private:
    value_t c_;
};

} // namespace expr
} // namespace ppl
