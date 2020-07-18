#include <array>
#include <autoppl/autoppl.hpp>

int main()
{
    std::array<double, 1000> mu_samples, sigma_samples;

    ppl::Data<double, ppl::vec> x {1.0, 1.5, 1.7, 1.2, 1.5};
    ppl::Param<double> mu {mu_samples.data()};
    ppl::Param<double> sigma {sigma_samples.data()};

    auto model = (
        mu |= ppl::normal(0., 3.),
        sigma |= ppl::uniform(0., 2.),
        x |= ppl::normal(mu, sigma)
    );

    ppl::nuts(model);

    double mu_mean = std::accumulate(
            mu_samples.begin(), mu_samples.end(), 0.) 
            / mu_samples.size();
    std::cout << "mu average: " 
              << mu_mean
              << std::endl;

    double sigma_mean = std::accumulate(
            sigma_samples.begin(), sigma_samples.end(), 0.) 
            / sigma_samples.size();
    std::cout << "sigma average: " 
              << sigma_mean
              << std::endl;
}
