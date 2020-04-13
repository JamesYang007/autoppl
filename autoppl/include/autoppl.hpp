#pragma once

namespace ppl {

inline int fib(int n)
{
    if (n <= 1) return 1;
    else return fib(n-1) + fib(n-2);
}

} // namespace autoppl
