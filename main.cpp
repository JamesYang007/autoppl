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
    std::array<double, 10000> storage;

    cout << ppl::normal(0.0, 1.0).pdf(5) << endl;

    auto y = ppl::variable(3.);
    auto x = ppl::variable(2.);

    auto z = ppl::variable({2, 3, 4});

    for (auto it = z.begin(); it != z.end(); it++) {
        cout << *it << endl;
    }

    std::vector vec = {1, 2, 3};
    auto a = ppl::variable(vec);

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
