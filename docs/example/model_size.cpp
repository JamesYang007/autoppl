#include <autoppl/autoppl.hpp>
#include <array>
#include <iostream>

void simple_model()
{
    // simple model size
    ppl::Data<double> X;
    ppl::Param<double> m;
    auto model = (
        m |= ppl::uniform(-1., 1.),
        X |= ppl::normal(m, 1.)
    );

    std::cout << "Model:\n"
              << "m ~ Uniform(-1, 1)\n"
              << "X ~ Normal(m, 1)\n"
              << std::endl;

    std::cout << "Size of model: " 
              << sizeof(model) << std::endl;
}

void complex_model()
{
    // complicated model size
    ppl::Data<double> X;
    std::array<ppl::Param<double>, 6> theta;
    auto model = (
        theta[0] |= ppl::uniform(-1., 1.),
        theta[1] |= ppl::uniform(theta[0], theta[0] + 2.),
        theta[2] |= ppl::normal(theta[1], theta[0] * theta[0]),
        theta[3] |= ppl::normal(-2., 1.),
        theta[4] |= ppl::uniform(-0.5, 0.5),
        theta[5] |= ppl::normal(theta[2] + theta[3], theta[4]),
        X |= ppl::normal(theta[5], 1.)
    );

    std::cout << "Model:\n"
              << "theta[0] |= ppl::uniform(-1., 1.),\n"
              << "theta[1] |= ppl::uniform(theta[0], theta[0] + 2.),\n"
              << "theta[2] |= ppl::normal(theta[1], theta[0] * theta[0]),\n"
              << "theta[3] |= ppl::normal(-2., 1.),\n"
              << "theta[4] |= ppl::uniform(-0.5, 0.5),\n"
              << "theta[5] |= ppl::normal(theta[2] + theta[3], theta[4]),\n"
              << "X |= ppl::normal(theta[5], 1.)"
              << std::endl;

    std::cout << "Size of model: " 
              << sizeof(model) << std::endl;
}

int main()
{
    simple_model();
    complex_model();
    return 0;
}
