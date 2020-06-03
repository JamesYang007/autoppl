#include <chrono>
#include <autoppl/expr_builder.hpp>

namespace ppl {

void update_time()
{
    auto begin = std::chrono::system_clock::now();
    // routine to time ...
    auto end = std::chrono::system_clock::now();
    size_t dur = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - begin).count();
    // ...
    static_cast<void>(dur);
}

} // namespace ppl

int main()
{
    ppl::update_time();
    return 0;
}
