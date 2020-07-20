#pragma once
#include <string>
#include <iostream>
#include <iomanip>

namespace ppl {

namespace util {

/**
 * ProgressLogger prints a visual progress bar counting towards 100% completion
 * of an algorithm or task (really, any for loop). max is the value being counted
 * towards, e.g. the upper-bound of the for-loop, and name is a string to print
 * alongside the progress bar. This prints directly to stdout, and no print statements
 * should be made in the intervening period between calls to printProgress.
 */
struct ProgressLogger {

    ProgressLogger(size_t max, const std::string & name, std::ostream& os = std::cout)
    : _max(max), _name(name), _os(os)
    {};

    /**
     * When the logger goes out of scope, append a new line
     */
    ~ProgressLogger(){
        _os << std::endl;
    }

    void printProgress(size_t step) {
        step = step+1;
        if (step % (_max / 100) == 0) {
            int percent = static_cast<int>(
                                           static_cast<double>(step) /
                                           (static_cast<double>(_max) / 100.)
            );
            _os << '\r' << _name << " ["
                << std::string(percent, '=')
                << std::string(100 - percent, ' ')
                << "] (" << std::setw(2)
                << percent
                << "%)" << std::flush;
        }
    }

private:
    size_t _max;  // maximum vaue to count progress towards
    std::string _name;  // name of the algorithm being measured, printed with progress bar
    std::ostream& _os; // output stream to print to
};


} // namespace util

} // namespace ppl
