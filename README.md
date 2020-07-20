# AutoPPL
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
[![Build Status](https://travis-ci.org/JamesYang007/autoppl.svg?branch=master)](https://travis-ci.org/JamesYang007/autoppl)
[![Coverage Status](https://coveralls.io/repos/github/JamesYang007/autoppl/badge.svg?branch=master)](https://coveralls.io/github/JamesYang007/autoppl?branch=master)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
    - [Distributions](#distributions)
    - [Sampling Algorithms](#sampling-algorithms)
- [Design Choices](#design-choices)
    - [Intuitive Model Specification](#intuitive-model-specification)
    - [Efficient Memory Usage](#efficient-memory-usage)
    - [High-performance Inference Methods](#high-performance-inference-methods)
- [Installation (dev)](#installation-dev)
- [Examples](#examples)
    - [Sampling from Joint Distribution](#sampling-from-joint-distribution)
    - [Sampling Posterior Mean and Standard Deviation](#sampling-posterior-mean-and-standard-deviation)
    - [Bayesian Linear Regression](#bayesian-linear-regression)
- [Benchmarks](#benchmarks)
    - [Bayesian Linear Regression](#benchmarks-bayesian-linear-regression)
    - [Gaussian Model](#gaussian-model)
- [Contributors](#contributors)
- [Third Party Tools](#third-party-tools)

## Overview

AutoPPL is a C++ template library providing high-level support for probabilistic programming.
Using operator overloading and expression templates, AutoPPL provides a 
generic framework for specifying probabilistic models and applying inference algorithms.

The library is still at an experimental stage.

## Features

### Distributions
- [Uniform](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))
- [Multivariate Normal (up to diagonal covariance matrix)](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
- [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution)

### Sampling Algorithms
- [Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)
- [No-U-Turn Sampler (NUTS)](http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf)

See [examples](#examples) for more detail on how to use these sampling algorithms.

## Design Choices

### Intuitive Model Specification 

In Bayesian statistics, the first step in performing probabilistic inference is to 
specify a generative model for the data of interest. For example, in mathematical notation, 
a Gaussian model could look like the following:
```
X ~ Normal(theta, I)
theta ~ Uniform(-1, 1)
```

Given its simplicity and expressiveness, we wanted to mimic this compact notation as much 
as possible for users of our library.

For the model specified above, the code in AutoPPL could look like the following:
```cpp
ppl::Data<double, ppl::vec> X({...});   // load data
ppl::Param<double> theta;               // define scalar parameter
auto model = (                          // specify model
    theta |= ppl::uniform(-1., 1.),
    X |= ppl::normal(theta, 1.)
);
```

A `ppl::Param<T>` represents a scalar parameter in our model, 
while `ppl::Data<T, ppl::vec>` holds observed data. 
The model can be specified by listing expressions 
that use `operator|=` to associate a variable with a distribution.

### Efficient Memory Usage

We make the assumption that users are able to specify the probabilistic model at compile-time.
As a result, AutoPPL can construct all model expressions at compile-time
simply from the type information.

A model object makes no heap-allocations and is minimal in size.
It is simply a small, contiguous slab of memory representing the binary tree.
The `model` object in the [previous section](#intuitive-model-specification)
is about `120 bytes` on `x86_64-apple-darwin17.7.0` using `clang-11.0.3`.

For a more complicated model such as the following:
```cpp
ppl::Data<double> X;
ppl::Param<double, ppl::vec> theta(6);
auto model = (
    theta[0] |= ppl::uniform(-1., 1.),
    theta[1] |= ppl::uniform(theta[0], theta[0] + 2.),
    theta[2] |= ppl::normal(theta[1], theta[0] * theta[0]),
    theta[3] |= ppl::normal(-2., 1.),
    theta[4] |= ppl::uniform(-0.5, 0.5),
    theta[5] |= ppl::normal(theta[2] + theta[3], theta[4]),
    X |= ppl::normal(theta[5], 1.)
);
```

The size of the model is `616 bytes` on the same machine and compiler.

A model expression simply references the variables used in the expression
such as `theta[0], theta[1], ..., theta[5], X`, i.e. it does not copy any data.

### High-performance Inference Methods

Users interface with the inference methods via model expressions and other
configuration parameters for that particular method.
Hence, the inference algorithms are completely general and work with any model
so long as the model expression is properly constructed.
Due to the statically-known model specification, algorithms
have opportunities to make compile-time optimizations.
See [Benchmarks](#benchmarks) for more detail. 

We were largely inspired by STAN and followed their 
[reference](https://mc-stan.org/docs/2_23/reference-manual/index.html)
and also their
[implementation](https://github.com/stan-dev/stan) 
to make statistical optimizations such as 
computing ESS, performing adaptations, and stabilizing sampling algorithms.
However, our library works quite differently underneath, especially
with automatic differentiation and handling model expressions.

## Installation (dev)

First, clone the repository:

```
git clone --recurse-submodules https://github.com/JamesYang007/autoppl
```

To build and run tests, run the following:
```
./setup.sh
./clean-build.sh debug
cd build/debug
ctest
```

## Examples

### Sampling from Joint Distribution

Although AutoPPL was designed to perform inference on posterior distributions,
one can certainly use it to sample from any joint distribution defined by the priors and conditional distributions.
As an example, we can sample `1000` points from a 
standard normal distribution using Metropolis-Hastings in the following way:

```cpp
std::array<double, 1000> samples;
ppl::Param<double> theta {samples.data()};
auto model = (
    theta |= ppl::normal(0., 1.)
);
ppl::mh(model, 1000);
```

The scalar parameter `theta` is bound to the storage `samples` at construction.
After calling `mh`, the `1000` samples are placed into `samples`.

In general, so long as the joint PDF is known, 
or equivalently and more commonly if the conditional and prior PDFs are known,
one can sample from the distribution.

As another example, we may sample from a more complicated joint distribution:
```cpp
std::array<double, 1000> theta1_samples;
std::array<double, 1000> theta2_samples;
ppl::Param<double> theta1 {theta1_samples.data()};
ppl::Param<double> theta2 {theta2_samples.data()};
auto model = (
    theta1 |= ppl::uniform(-1., 1.),
    theta2 |= ppl::normal(theta1, 1.)
);
ppl::mh(model, 1000);
```

Here, `theta2` depends on `theta1` and hence defines
the conditional distribution `theta2 | theta1`.

### Sampling Posterior Mean and Standard Deviation

The following is an example of fitting a Gaussian model to some data.
We put a `Normal(0,3)` prior on the mean and `Uniform(0,2)` prior on the 
standard deviation.
While in the previous section, we used Metropolis-Hastings 
to demonstrate how to use it,
it is recommended to use the state-of-the-art NUTS sampler to sample
from the posterior distribution.

```cpp
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
```

By default, NUTS will run 1000 iterations each for warmup and actual sampling
(total of 2000 iterations).
The default adaptation method is to estimate the diagonal precision matrix.
If the user wishes to change the configuration, one must pass in a config object:

```cpp
NUTSConfig<...> config; // replace ... with ppl::unit_var or ppl::diag_var
config.n_samples = ...; // number of samples
config.warmup = ...;    // number of warmup iters
config.seed = ...;      // set seed (default: random)
config.max_depth = ...; // max depth of tree built during NUTS (default: 10)

ppl::nuts(model, config);
```

### Bayesian Linear Regression

This example covers ridge regression in a Bayesian setting.
We created a fictitious dataset consisting of `(x,y)` coordinates.
The true relationship is the following: `y = x + 1`.
By specifying two parameters for the weight and bias, we propose
the following probabilistic model:

```
y ~ Normal(x*w + b, 0.5)
w ~ Uniform(0, 2)
b ~ Uniform(0, 2)
```

In AutoPPL, we can write the following code and sample from the posterior:
```cpp
std::array<double, 1000> w_storage;
std::array<double, 1000> b_storage;

ppl::Data<double> x{2.4, 3.1, 3.6, 4, 4.5, 5.};
ppl::Data<double> y{3.5, 4, 4.4, 5.01, 5.46, 6.1};
ppl::Param<double> w {w_storage.data()};
ppl::Param<double> b {b_storage.data()};

auto model = (
        w |= ppl::uniform(0., 2.),
        b |= ppl::uniform(0., 2.),
        y |= ppl::normal(x * w + b, 0.5)
);

ppl::nuts(model);
```

## Benchmarks

In the following examples, we show benchmarks with STAN.

We list the benchmark settings for completion:
- Machine: x86_64-apple-darwin19.5.0
- CPU: 3.4 GHz Quad-Core Intel Core i5
- Compiler: Clang 11.0.3

### Bayesian Linear Regression <a name="benchmarks-bayesian-linear-regression"></a>

We collected a dataset regarding life expectancy released by WHO 
([source](https://www.kaggle.com/kumarajarshi/life-expectancy-who)).
After cleaning and extracting three predictors: "Alcohol", "HIV/AIDS", and "GDP",
the dataset consisted of 157 points.
We performed a Bayesian linear regression with this data and the following model:

```
y ~ Normal(X*w + b, s*s + 2.)
w ~ Normal(0., 5.)
b ~ Normal(0., 5.)
s ~ Uniform(0.5, 8.)
```

where `w` is a 3-dimensional parameter vector,
and `b` and `s` are scalar parameters.

Using the same dataset and model specification,
we performed NUTS to sample various number of samples.
We also set the number of chains and cores to 1 and 
adaptation method to diagonal precision matrix.

The following plots show benchmarks between
run-times and effective sample size (ESS) per second:

![](docs/figures/regression_benchmark_plot/runtime.png)
![](docs/figures/regression_benchmark_plot/ess_s.png)

The reported mean, standard deviation, and ESS values were 
almost identical in all cases,
which is not surprising since we used the same algorithm to estimate ESS
and perform NUTS.

The runtimes have similar log-like behavior,
but it is clear that STAN (dotted lines) takes far longer
in both sampling and warmup times.
As for ESS/s, upon comparing by colors (corresponding to a parameter)
between dotted (STAN) and solid (AutoPPL) lines,
we see that AutoPPL has uniformly larger ESS/s.
This difference quickly becomes larger as sample size grows.
From these plots and that sampling results were identical
show that the drastic difference in ESS/s is simply from faster
computations such as automatic differentation during NUTS sampling,
and how we interpret the model expression to create compile-time optimizations.

ESS was computed as outlined 
[here](https://mc-stan.org/docs/2_23/reference-manual/effective-sample-size-section.html).
We also made some adjustments to use Geyer's biased estimator for ESS
as in the current implementation of STAN
([source](https://github.com/stan-dev/stan/blob/525998129ea838ec685f1d1f65dc76063d0fd40d/src/stan/analyze/mcmc/compute_effective_sample_size.hpp)).

The following is the AutoPPL code for the model specification without data loading.
The full code can be found
[here](benchmark/regression_autoppl.cpp):

```cpp
auto X = ppl::make_data_view<ppl::mat>(X_data);
auto y = ppl::make_data_view<ppl::vec>(y_data); 
ppl::Param<double, ppl::vec> w(3);
ppl::Param<double> b;
ppl::Param<double> s;

arma::mat storage(num_samples, w.size() + b.size() + s.size());
for (size_t i = 0; i < w.size(); ++i) {
    w.storage(i) = storage.colptr(i);
}
b.storage() = storage.colptr(w.size());
s.storage() = storage.colptr(w.size() + b.size());

auto model = (s |= ppl::uniform(0.5, 8.),
              b |= ppl::normal(0., 5.),
              w |= ppl::normal(0., 5.),
              y |= ppl::normal(ppl::dot(X, w) + b, s * s + 2.));

NUTSConfig<> config;
config.warmup = num_samples;
config.n_samples = num_samples;
auto res = ppl::nuts(model, config);
```

### Gaussian Model

We generated 1000 values from standard normal distribution to form our data.
Our model is defined as follows:

```
y ~ Normal(l1 + l2, s) 
l1 ~ Normal(0., 10.)
l2 ~ Normal(0., 10.)
s ~ Uniform(0., 20.)
```

where all parameters are scalar.

The benchmark configurations are exactly the same as in the
[previous section](#benchmarks-bayesian-linear-regression).

The following plots show benchmarks between
run-times and effective sample size (ESS) per second:

![](docs/figures/gaussian_benchmark_plot/runtime.png)
![](docs/figures/gaussian_benchmark_plot/ess_s.png)

We note that both STAN and AutoPPL outputted almost identical
means, standard deviation, and ESS.

The runtimes have a similar linear trend,
and it is clear that STAN (dotted lines) takes far longer
in both sampling and warmup times.
The ESS/s for `l1` and `l2` overlap completely (red and blue) in both STAN and AutoPPL
and this is expected since they are symmetric in the model specification.
With the exception of the two smallest sample sizes (100, 500),
ESS/s is fairly constant as sample size varies.
It is quite evident that AutoPPL (solid) has a larger ESS/s by a significant factor.

The reason for this difference is in how we handle expressions
where data vector elements are iid (independent and identically distributed).
For most distributions, especially those that are in some exponential family,
they can be highly optimized in iid settings to perform quicker differentiation.
However, it is worth noting that this optimization does not apply when
the data are simply independent but not identically distributed
(as in the [linear regression](#benchmarks-bayesian-linear-regression) case),
or when the variable is a parameter, not data.

The following is the AutoPPL code without data generation.
The full code can be found 
[here](benchmark/normal_two_prior_distribution.cpp).

```cpp
ppl::Param<double> l1, l2, s;

auto model = (
    s |= ppl::uniform(0.0, 20.0),
    l1 |= ppl::normal(0.0, 10.0),
    l2 |= ppl::normal(0.0, 10.0),
    y |= ppl::normal(l1 + l2, s)
);

arma::mat storage(n_samples, 3);
l1.storage() = storage.colptr(0);
l2.storage() = storage.colptr(1);
s.storage() = storage.colptr(2);

ppl::NUTSConfig<> config;
config.n_samples = n_samples;
config.warmup = n_samples;

auto res = ppl::nuts(model, config);
```

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://jacobaustin123.github.io/"><img src="https://avatars0.githubusercontent.com/u/28993550?v=4" width="100px;" alt=""/><br /><sub><b>Jacob Austin</b></sub></a><br /><a href="https://github.com/JamesYang007/autoppl/commits?author=jacobaustin123" title="Code">💻</a> <a href="#design-jacobaustin123" title="Design">🎨</a> <a href="https://github.com/JamesYang007/autoppl/commits?author=jacobaustin123" title="Documentation">📖</a></td>
    <td align="center"><a href="http://jenny-chen.net"><img src="https://avatars0.githubusercontent.com/u/13106682?v=4" width="100px;" alt=""/><br /><sub><b>Jenny Chen</b></sub></a><br /><a href="https://github.com/JamesYang007/autoppl/commits?author=jenchen1398" title="Code">💻</a> <a href="#design-jenchen1398" title="Design">🎨</a></td>
    <td align="center"><a href="https://github.com/lucieleblanc"><img src="https://avatars3.githubusercontent.com/u/14223323?v=4" width="100px;" alt=""/><br /><sub><b>lucieleblanc</b></sub></a><br /><a href="https://github.com/JamesYang007/autoppl/commits?author=lucieleblanc" title="Code">💻</a> <a href="#design-lucieleblanc" title="Design">🎨</a></td>
    <td align="center"><a href="http://lancaster.ac.uk/~ludkinm/"><img src="https://avatars3.githubusercontent.com/u/28777642?v=4" width="100px;" alt=""/><br /><sub><b>Matt Ludkin</b></sub></a><br /><a href="https://github.com/JamesYang007/autoppl/commits?author=ludkinm" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## Third Party Tools

Many thanks to the following third party tools used in this project:
- [Armadillo](http://arma.sourceforge.net/): matrix library used for inference algorithms.
- [Clang](https://clang.llvm.org/): one of the main compilers used.
- [CMake](https://cmake.org/): build system.
- [Coveralls](https://coveralls.io/): check test coverage.
- [Cpp Coveralls](https://github.com/eddyxu/cpp-coveralls): check test coverage specifically for C++ code.
- [FastAD](https://github.com/JamesYang007/FastAD): automatic differentiation library.
- [GCC](https://gcc.gnu.org/): one of the main compilers used.
- [Google Benchmark](https://github.com/google/benchmark): benchmark library algorithms.
- [GoogleTest](https://github.com/google/googletest): unit/integration-tests.
- [Travis CI](https://travis-ci.org/): continuous integration for Linux using GCC.
- [Valgrind](http://valgrind.org/): check memory leak and errors.

