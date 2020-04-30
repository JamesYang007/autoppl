#include <array>
#include <autoppl/autoppl.hpp>

int main()
{
    std::array<double, 1000> samples;
    ppl::Param<double> theta {samples.data()};
    auto model = (
        theta |= ppl::normal(0., 1.)
    );
    ppl::mh(model, 1000);

    double mean = std::accumulate(
            samples.begin(), samples.end(), 0.) 
            / samples.size();
    std::cout << "Sample average: " 
              << mean
              << std::endl;

    return 0;
}
