#include <array>
#include <autoppl/autoppl.hpp>

int main()
{
    std::array<double, 1000> theta1_samples;
    std::array<double, 1000> theta2_samples;
    ppl::Param<double> theta1 {theta1_samples.data()};
    ppl::Param<double> theta2 {theta2_samples.data()};
    auto model = (
        theta1 |= ppl::uniform(-1., 1.),
        theta2 |= ppl::normal(theta1, 1.)
    );

    ppl::nuts(model);

    double theta1_mean = std::accumulate(
            theta1_samples.begin(), theta1_samples.end(), 0.) 
            / theta1_samples.size();
    double theta2_mean = std::accumulate(
            theta2_samples.begin(), theta2_samples.end(), 0.) 
            / theta2_samples.size();

    std::cout << "Theta1 sample average: " 
              << theta1_mean
              << std::endl;
    std::cout << "Theta2 sample average: " 
              << theta2_mean
              << std::endl;

    return 0;
}
