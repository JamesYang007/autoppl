#pragma once
#include <autoppl/util/traits/dist_expr_traits.hpp>

#define PPL_DIST_SHAPE_MISMATCH \
    "Unsupported variable and/or distribution parameter shapes. "
#define PPL_PDF_INVOCABLE \
    "Log-pdf and pdf functors must be invocable with a single size_t argument. "

namespace ppl {
namespace expr {

/**
 * Computes joint log pdf defined by size number of independent variables.
 * log_pdf(i) computes the log pdf of ith variable. 
 */
template <class LogPDFType>
inline constexpr auto log_pdf_indep(LogPDFType&& log_pdf,
                                    size_t size) 
{
    static_assert(std::is_invocable_v<LogPDFType, size_t>,
                  PPL_PDF_INVOCABLE);
    using dist_value_t = std::decay_t<
        decltype(log_pdf(std::declval<size_t>()))>;
    dist_value_t value = 0.0;
    for (size_t i = 0ul; i < size; ++i) {
        value += log_pdf(i);
    }
    return value;
}

/**
 * Computes joint pdf defined by size number of independent variables.
 * pdf(i) computes the pdf of ith variable. 
 */
template <class PDFType>
inline constexpr auto pdf_indep(PDFType&& pdf,
                                size_t size) 
{
    static_assert(std::is_invocable_v<PDFType, size_t>,
                  PPL_PDF_INVOCABLE);
    using dist_value_t = std::decay_t<
        decltype(pdf(std::declval<size_t>()))>;
    dist_value_t value = 1.0;
    for (size_t i = 0ul; i < size; ++i) {
        value *= pdf(i);
    }
    return value;
}

} // namespace expr
} // namespace ppl
