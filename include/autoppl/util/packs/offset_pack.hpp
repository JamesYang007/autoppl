#pragma once
#include <cstdint>
#include <cstddef>

namespace ppl {
namespace util {

struct OffsetPack
{
    using index_t = uint32_t;
    index_t uc_offset = 0;       // unconstrained param offset
    index_t c_offset = 0;        // constrained param offset
    index_t v_offset = 0;        // visit count offset
    index_t tp_offset = 0;       // transformed param offset
};

} // namespace util
} // namespace ppl
