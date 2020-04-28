# AutoPPL

[![Build Status](https://travis-ci.org/JamesYang007/autoppl.svg?branch=master)](https://travis-ci.org/JamesYang007/autoppl)
[![Coverage Status](https://coveralls.io/repos/github/JamesYang007/autoppl/badge.svg?branch=master)](https://coveralls.io/github/JamesYang007/autoppl?branch=master)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)

## Overview

AutoPPL is a C++ library providing high-level support for probabilistic programming.

## Installation

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

Sampling from a normal distribution:

```cpp
std::array<double, 1000> mu_arr, sigma_arr;

ppl::Data<double> x {1.0, 1.5, 1.7, 1.2, 1.5};
ppl::Param<double> mu {mu_arr.data()}, sigma {sigma_arr.data()};

auto model = (
  mu |= ppl::normal(0., 3.),
  sigma |= ppl::uniform(0., 2.),
  x |= ppl::normal(mu, sigma)
)

ppl::mh_posterior(model, 1000);
```

Bayesian linear regression:

```cpp
std::array<double, 10000> w_storage;
std::array<double, 10000> b_storage;

ppl::Data<double> x {2.5, 3, 3.5, 4, 4.5, 5};
ppl::Data<double> y {3.5, 4, 4.5, 5, 5.5, 6.};
ppl::Param<double> w {w_storage.data()};
ppl::Param<double> b {b_storage.data()};

auto model = (
        w |= ppl::uniform(0, 2),
        b |= ppl::uniform(0, 2),
        y |= ppl::normal(x * w + b, 0.5)
);

auto log_pdf = model.log_pdf();
ppl::mh_posterior(model, 10000);
```
