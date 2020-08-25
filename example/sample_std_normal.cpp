#include <array>
#include <autoppl/autoppl.hpp>

int main()
{
    ppl::Param<double> theta;
    auto model = (
        theta |= ppl::normal(0., 1.)
    );

    ppl::MHConfig config;
    config.warmup = 1000;
    config.samples = 1000;
    auto res = ppl::mh(model, config);

    auto samples = res.cont_samples.col(0);

    double mean = std::accumulate(
            samples.data(), samples.data() + samples.size(), 0.) 
            / samples.size();
    std::cout << "Sample average: " 
              << mean
              << std::endl;

    return 0;
}
