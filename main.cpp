#include <array>
#include <autoppl/expr_builder.hpp>
#include <autoppl/algorithm/mh.hpp>
#include <cmath>
#include <iostream>
#include <random>
#include <numeric>
#include <vector>

using namespace std;

int main() {
    std::array<double, 1000> sigma_storage;


    //    cout << ppl::normal(0.0, 1.0).pdf(5) << endl;

    ppl::Variable<double> x {2.5};
    x.get_value(1);
    ppl::Variable<double> sigma {sigma_storage.data()};

    auto model = (
            sigma |= ppl::normal(0.5, 1.),
            x |= ppl::normal(2.5, sigma)
    );

    ppl::mh_posterior(model, 1000);

    double mean = 0;
    for (auto val : sigma_storage) {
        mean += val;
    }

    mean /= 1000;
    cout << mean << endl;

//    y.observe(3.0);
//    x.observe(2.0);
//
//    auto model = (
//            mu |= ppl::uniform(-2., 2.),
//            y |= ppl::normal(mu, 1.)
//    );
//
//    ppl::mh_posterior(model, 10000);
//
//    double mean = 0;
//    for (auto val : storage) {
//        mean += val;
//    }
//
//    mean /= 10000;
//
//    cout << mean << endl;
}
