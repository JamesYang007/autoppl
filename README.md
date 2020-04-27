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

```cpp
ppl::Variable<double> x {1.0, 1.5, 1.7, 1.2, 1.5};
ppl::Variable<double> mu, sigma;
auto model = (
  mu |= ppl::normal(0., 3.),
  sigma |= ppl::uniform(0., 2.),
  x |= ppl::normal(mu, sigma)
)

ppl::mh_posterior(model, 1000);
```
