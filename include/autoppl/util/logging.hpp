#pragma once
#include <string>
#include <iostream>
#include <iomanip>

namespace ppl {
namespace util {

/**
 * ProgressLogger prints a visual progress bar counting towards 100% completion
 * of an algorithm or task (really, any for loop). 
 *
 * This prints directly to the output stream provided in the constructor, 
 * and no print statements should be made in the period between calls to printProgress.
 *
 */

struct ProgressLogger 
{

    /**
     * Constructs a ProgressLogger object.
     *
     * @param   max     the value being counted towards,    
     *                  e.g. the upper-bound of the for-loop
     * @param   name    string to print alongside the progress bar
     */
    ProgressLogger(size_t max, 
                   const std::string& name, 
                   std::ostream& os = std::cout)
        : max_(max)
        , name_(name)
        , os_(os)
    {};

    /**
     * When the logger goes out of scope, append a new line
     */
    ~ProgressLogger() { os_ << std::endl; }

    void printProgress(size_t step) {
        ++step;
        size_t denom = max_ <= 100 ? max_ : 100;
        if (step % (max_ / denom) == 0) {
            int percent = static_cast<double>(step * 100) /
                            static_cast<double>(max_);
            os_ << '\r' << name_ << " ["
                << std::string(percent, '=')
                << std::string(100 - percent, ' ')
                << "] (" << std::setw(2)
                << percent
                << "%)" << std::flush;
        }
    }

private:
    size_t max_;        // maximum vaue to count progress towards
    std::string name_;  // name of the algorithm being measured, printed with progress bar
    std::ostream& os_;  // output stream to print to
};

} // namespace util
} // namespace ppl
