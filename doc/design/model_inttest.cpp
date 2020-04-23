#include "gtest/gtest.h"
#include <autoppl/expression/model.hpp>
#include <autoppl/expression/rv_tag.hpp>
#include <autoppl/expression/uniform.hpp>

namespace ppl {

template <class VectorType, class IndexType>
struct BracketNode
{
    VectorType v;
    IndexType i;
};

struct myvector
{

    rv_tag<double> operator[](rv_tag)
    {
        return rv
    }
    std::vector<rv_tags> v; // 3 things
};

template <class MuType, class SigType>
auto normal(const MuType& mu, const SigType& sig)
{
    Normal<MuType, SigType>(mu, sig);
}

TEST(dummy, dummy_test)
{
    double x_data = 2.3; // 1-sample data

    std::vector<double> sampled_theta_1(100);
    std::vector<double> sampled_theta_2(100);

    double* ptr;
    rv_tag<double, ...> x;
    rv_tag<double> theta_1(sampled_theta_1.data());
    rv_tag<double> theta_2(sampled_theta_2.data());

    std::vector<rv_tag<double>> v;
    std::for_each(..., ... , [](){v[i].set_sample_storage(&mat.row(i));});

    x.observe(x_data);

    x_1.observe(...);
    x_2.observe(...);

    auto model = (
        mu |= uniform(-10000, 10000),
        y |= uniform({1,2,3})   // 
        x_1 |= normal(mu[y], 1),
        x_2 |= normal(mu[y], 1),
    );

    x.observe(...);

    rv_tag<double> var, mu, x;
    auto normal_model = (
        var |= normal(0,1),
        mu |= normal(1,5),
        x |= normal(mu, var)
    );

    std::vector<double> var_storage(1000);
    std::vector<double> mu_storage(1000);

    var.set_storage(var_storage.data());
    mu.set_storage(mu_storage.data());

    metropolis_hastings(model, 1000, 400);

    auto gmm_model = (
        mu |=    
    );

    std::vector<rv_tag<double>> vec(model.param_num);
    model.bind_storage(vec.begin(), vec.end(), ...);
    model.pdf();

    metropolis_hastings(model, 100);

    std::vector<double> sampled_theta_1_again(1000);
    std::vector<double> sampled_theta_2_again(1000);

    theta_1.set_storage(sampled_theta_1_again.data());
    theta_2.set_storage(sampled_theta_2_again.data());

    metropolis_hastings(model, 1000);







    auto model = (
        w |= normal(0,1),
        y |= normal(w*x, 1)
    )
    metropolis_hastings(modeli)
}

} 
