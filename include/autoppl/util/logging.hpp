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
    ProgressLogger(size_t max, const std::string & name) 
        : _max(max), _name(name) 
    {};

    void printProgress(size_t step) {
        if (step % (_max / 100) == 0) {
            int percent = static_cast<int>(
                    static_cast<double>(step) / 
                    (static_cast<double>(_max) / 100.)
                    );
            std::cout << '\r' << _name << " [" 
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
};


} // namespace util

} // namespace ppl
