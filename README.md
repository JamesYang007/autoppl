# AutoPPL

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
- [Contributors](#contributors)
- [Third Party Tools](#third-party-tools)

## Overview

AutoPPL is a C++ template library providing high-level support for probabilistic programming.
Using operator overloading and expression templates, AutoPPL provides a 
generic framework to specify probabilistic models and apply inference algorithms on them.

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

In hierarchical Bayesian statistics, the first step before doing any inference
is to specify a probabilistic model for the data of interest.
For example, in mathematical notation, a Gaussian model could look like the following:
```
X ~ N(mu, I)
mu ~ U(-1, 1)
```

Given its simplicity and expressiveness,
we wanted to mimic this compact notation as much as possible for the users.

For the model specified above, the code in AutoPPL could look like the following:
```cpp
ppl::Data<double> X;
ppl::Param<double> mu;
auto model = (
    mu |= ppl::uniform(-1., 1.),
    X |= ppl::normal(mu, 1)
);
```

### Efficient Memory Usage

We make the assumption that users are able to specify the probabilistic model at compile-time.
As a result, AutoPPL can construct all model expressions at compile-time
simply from the type information and using expression templates.
A model expression is simply a binary tree that relates variables with 
distributions, which can potentially depend on other variables.

As an example, our model expression such as `model` in the previous 
[section](#intuitive-model-specification) is a compile-time constructed (binary) tree 
storing information about the relationship between `m, X` and their (conditional) distributions.

A model object makes no heap-allocations and is extremely minimal in size.
It is simply a small, contiguous slab of memory representing the binary tree.
The `model` object in the previous [section](#intuitive-model-specification)
is about `48 bytes` on `x86_64-apple-darwin17.7.0` using `clang-10`.

For a more complicated model such as the following:
```cpp
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
```

The size of the model is `192 bytes` on the same machine and compiler.

A model expression simply references the variables used in the expression
such as `theta[0], theta[1], ..., theta[5], X`.
The model does not store any data, but rather only serves to relate variables.

### High-performance Inference Methods

Users interface with the inference methods via model expressions and other
configuration parameters for that particular method.
Hence, the inference algorithms are completely generalized to work for any model,
so long as the model expression is properly constructed.
Moreover, due to the statically-known model specification, algorithms
have opportunities to make compile-time optimizations.
See [Examples](#examples) for more detail.

(TODO: include benchmark results)

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

Although `AutoPPL` was designed to perform inference on posterior distributions,
one can certainly use it to simply sample from any joint distribution.
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

Note that the parameter `theta` is bound to the storage `samples` at construction.
After calling `mh`, the `1000` samples are placed into `samples.`

In general, so long as the joint PDF is known, 
or equivalently and more commonly if the conditional PDFs are known,
one can sample from the distribution.

As another example, we may sample from such joint distribution:
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

### Sampling Posterior Mean and Standard Deviation

The following is an example of fitting a Gaussian model to some data.
We put a Normal(0,3) prior on the mean and Uniform(0,2) prior on the 
standard deviation.
It is recommended to use the state-of-the-art NUTS sampler to sample
from a posterior distribution.

```cpp
std::array<double, 1000> mu_samples, sigma_samples;

ppl::Data<double> x {1.0, 1.5, 1.7, 1.2, 1.5};
ppl::Param<double> mu {mu_samples.data()};
ppl::Param<double> sigma {sigma_samples.data()};

auto model = (
    mu |= ppl::normal(0., 3.),
    sigma |= ppl::uniform(0., 2.),
    x |= ppl::normal(mu, sigma)
);

size_t warmup = 1000;
size_t n_samples = 1000;
size_t n_adapt = 1000;
ppl::nuts(model, warmup, n_samples, n_adapt);
```

### Bayesian Linear Regression

This example covers ordinary linear regression in a Bayesian setting.
We created a fictitious dataset consisting of `(x,y)` coordinates.
The true relationship is the following: `y = x + 1`.
By specifying two parameters for the weight and bias, we propose
the following probabilistic model:
```
y ~ N(wx + b, 0.5)
w ~ U(0, 2)
b ~ U(0, 2)
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
        w |= ppl::uniform(0, 2),
        b |= ppl::uniform(0, 2),
        y |= ppl::normal(x * w + b, 0.5)
);

size_t warmup = 1000;
size_t n_samples = 1000;
size_t n_adapt = 1000;
ppl::nuts(model, warmup, n_samples, n_adapt);
```

## Contributors

| **James Yang** | **Jacob Austin** | **Jenny Chen** | **Lucie Le Blanc** |
| :---: | :---: | :---: | :---: |
| [![JamesYang007](https://avatars3.githubusercontent.com/u/5008832?s=150&v=4)](https://github.com/JamesYang007) | [![jacobaustin123](https://avatars1.githubusercontent.com/u/28993550?s=150&u=151a97ac00d11e39ab3c0dbb8920dad934177d06&v=4)](https://github.com/jacobaustin123) | [![jenchen1398](https://avatars2.githubusercontent.com/u/13106682?s=150&u=926a625662c64366b66355dbdfc7b02f21dd4c91&v=4)](https://github.com/jenchen1398) | [![lucieleblanc](https://avatars1.githubusercontent.com/u/14223323?s=20&v=4)](https://github.com/lucieleblanc) |
| <a href="http://github.com/JamesYang007" target="_blank">`github.com/JamesYang007`</a> | <a href="https://github.com/jacobaustin123" target="_blank">`github.com/jacobaustin123`</a> | <a href="https://github.com/jenchen1398" target="_blank">`github.com/jenchen1398`</a> | <a href="https://github.com/lucieleblanc" target="_blank">`github.com/lucieleblanc`</a> |

## Third Party Tools

Many thanks to the following third party tools used in this project:
- [Clang](https://clang.llvm.org/): one of the main compilers used.
- [CMake](https://cmake.org/): build system.
- [Coveralls](https://coveralls.io/): check test coverage.
- [Cpp Coveralls](https://github.com/eddyxu/cpp-coveralls): check test coverage specifically for C++ code.
- [GCC](https://gcc.gnu.org/): one of the main compilers used.
- [Google Benchmark](https://github.com/google/benchmark): benchmark library algorithms.
- [GoogleTest](https://github.com/google/googletest): unit/integration-tests.
- [Travis CI](https://travis-ci.org/): continuous integration for Linux using GCC.
- [Valgrind](http://valgrind.org/): check memory leak and errors.

