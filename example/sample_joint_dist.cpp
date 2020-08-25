#include <array>
#include <autoppl/autoppl.hpp>

int main()
{
    ppl::Param<double> theta1;
    ppl::Param<double> theta2;
    auto model = (
        theta1 |= ppl::uniform(-1., 1.),
        theta2 |= ppl::normal(theta1, 1.)
    );

    auto res = ppl::nuts(model);

    auto theta1_samples = res.cont_samples.col(0);
    auto theta2_samples = res.cont_samples.col(1);

    double theta1_mean = std::accumulate(
            theta1_samples.data(), theta1_samples.data() + theta2_samples.size(), 0.) 
            / theta1_samples.size();
    double theta2_mean = std::accumulate(
            theta2_samples.data(), theta2_samples.data() + theta2_samples.size(), 0.) 
            / theta2_samples.size();

    std::cout << "Theta1 sample average: " 
              << theta1_mean
              << std::endl;
    std::cout << "Theta2 sample average: " 
              << theta2_mean
              << std::endl;

    return 0;
}
