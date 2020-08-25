#include <array>
#include <autoppl/autoppl.hpp>

int main()
{
    ppl::Data<double, ppl::vec> x(5);
    ppl::Param<double> mu;
    ppl::Param<double> sigma;

    x.get() << 1.0, 1.5, 1.7, 1.2, 1.5;

    auto model = (
        mu |= ppl::normal(0., 3.),
        sigma |= ppl::uniform(0., 2.),
        x |= ppl::normal(mu, sigma)
    );

    auto res = ppl::nuts(model);

    auto mu_samples = res.cont_samples.col(0);
    auto sigma_samples = res.cont_samples.col(1);

    double mu_mean = std::accumulate(
            mu_samples.data(), mu_samples.data() + mu_samples.size(), 0.) 
            / mu_samples.size();
    std::cout << "mu average: " 
              << mu_mean
              << std::endl;

    double sigma_mean = std::accumulate(
            sigma_samples.data(), sigma_samples.data() + sigma_samples.size(), 0.) 
            / sigma_samples.size();
    std::cout << "sigma average: " 
              << sigma_mean
              << std::endl;
}
