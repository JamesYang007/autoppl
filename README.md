# AutoPPL
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
[![Build Status](https://travis-ci.org/JamesYang007/autoppl.svg?branch=master)](https://travis-ci.org/JamesYang007/autoppl)
[![Coverage Status](https://coveralls.io/repos/github/JamesYang007/autoppl/badge.svg?branch=master)](https://coveralls.io/github/JamesYang007/autoppl?branch=master)

## Table of Contents

- [Overview](#overview)
    - [Who should use AutoPPL?](#who-should-use-autoppl)
    - [How is AutoPPL different from existing PPLs like STAN and PyMC3?](#how-is-autoppl-different-from-existing-ppls-like-stan-and-pymc3)
- [Design Choices](#design-choices)
    - [Intuitive Model Specification](#intuitive-model-specification)
    - [Efficient Memory Usage](#efficient-memory-usage)
    - [High-performance Inference Methods](#high-performance-inference-methods)
- [Installation](#installation)
- [Quick Guide](#quick-guide)
    - [Variable](#variable)
    - [Variable Expression](#variable-expression)
    - [Constraint](#constraint)
    - [Distribution Expression](#distribution-expression)
    - [Model Expression](#model-expression)
    - [Transformed Parameters](#transformed-parameters)
    - [Program Expression](#program-expression)
    - [Sampling Algorithms](#sampling-algorithms)
- [Examples](#examples)
    - [Sampling from Joint Distribution](#sampling-from-joint-distribution)
    - [Sampling Posterior Mean and Standard Deviation](#sampling-posterior-mean-and-standard-deviation)
    - [Bayesian Linear Regression](#bayesian-linear-regression)
    - [Stochastic Volatility](#stochastic-volatility)
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

### Who should use AutoPPL?

The goal of this project is to provide a framework for practitioners, students, and researchers.
It is often desired to have a framework for specifying a probabilistic model separate from any inference algorithms.
While AutoPPL does provide a few inference algorithms such as NUTS and Metropolis-Hastings,
it allows users to write their _own_ sampling algorithms and even add new distributions.

### How is AutoPPL different from existing PPLs like STAN and PyMC3?

AutoPPL can be thought of as a hybrid of STAN and PyMC3.
It is similar to STAN in that it is extremely optimized for high performance (see [Benchmarks](#benchmarks))
and it uses much of the same logic discussed in the STAN reference guide.
It is similar to PyMC3 in that it is a library rather than a separate domain-specific language.

However, it is unique in that it is purely a C++ library.
While STAN requires the user to write STAN code, which gets translated into C++ code by the STAN compiler
and then compiled into a binary, AutoPPL just requires the user to directly write C++ code.
Some benefits include the following:
- eliminates the extra layer of abstraction to a separate domain-specific language
- users can use native C++ tools to clean and prepare data, and also examine the posterior samples
- easily extend the library such as adding new distributions or a sampling algorithm

In the future, we plan to provide Python and R bindings such that 
the user can write a C++ function using AutoPPL that defines and samples from the model
and export the Python/R binding to examine the posterior samples using a more comfortable, scripting language.

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
Data<double, vec> X({...});   // load data
Param<double> theta;          // define scalar parameter
auto model = (                // specify model
    theta |= uniform(-1., 1.),
    X |= normal(theta, 1.)
);
```

### Efficient Memory Usage

We make the assumption that users are able to specify the probabilistic model at compile-time.
As a result, AutoPPL can construct all model expressions at compile-time simply from the type information. 
A model object makes no heap-allocations (with the exception of `ppl::for_each`) and is minimal in size.
It is simply a small, contiguous slab of memory representing the binary tree.
The `model` object in the [previous section](#intuitive-model-specification)
is about `88 bytes` on `x86_64-apple-darwin17.7.0` using `clang-11.0.3`.

For a more complicated model such as the following:
```cpp
Data<double> X;
std::array<Param<double>, 6> theta;
auto model = (
    theta[0] |= uniform(-1., 1.),
    theta[1] |= uniform(theta[0], theta[0] + 2.),
    theta[2] |= normal(theta[1], theta[0] * theta[0]),
    theta[3] |= normal(-2., 1.),
    theta[4] |= uniform(-0.5, 0.5),
    theta[5] |= normal(theta[2] + theta[3], theta[4]),
    X |= normal(theta[5], 1.)
);
```
The size of the model is `440 bytes` on the same machine and compiler.

A model expression simply references the variables used in the expression
such as `theta[0], theta[1], ..., theta[5], X`, i.e. it does not copy any data or values.

### High-performance Inference Methods

Users interface with the inference methods via model expressions and other
configuration parameters for that particular method.
Hence, the inference algorithms are completely general and work with any model
so long as the model expression is properly constructed.
Due to the statically-known model specification, algorithms
have opportunities to make compile-time optimizations.
See [Benchmarks](#benchmarks) for performance comparisons with STAN. 

We were largely inspired by STAN and followed their 
[reference](https://mc-stan.org/docs/2_23/reference-manual/index.html)
and also their
[implementation](https://github.com/stan-dev/stan) 
to compute ESS, perform adaptations, and stabilize sampling algorithms.
However, our library works very differently underneath, especially
with automatic differentiation and handling model expressions.

## Installation

First, clone the repository:
```
git clone https://github.com/JamesYang007/autoppl ~/autoppl
```

The following are the dependencies:
- [Eigen3.3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [FastAD](https://github.com/JamesYang007/FastAD)

We provide a shell script for Mac and Linux that automatically installs these libraries locally in `lib`:
```
cd ~/autoppl
./setup.sh
```

Since AutoPPL is a template library, there is nothing to build!
Just pass the compiler flag `-I<path>` when building your program
to include the path to `include` inside cloned directory,
`include` to Eigen3.3, and `include` to FastAD.
If you ran `./setup.sh`, the paths for the latter two are
`lib/FastAD/libs/eigen-3.3.7/build/include` and
`lib/FastAD/build/include` (relative to cloned directory).

For CMake users, they can follow these steps:
```
./clean-build.sh release -DCMAKE_INSTALL_PREFIX=.. -DAUTOPPL_ENABLE_TEST=OFF
cd build/release
make install
```
This will simply set the CMake variable `CMAKE_INSTALL_PREFIX` to the `build` directory and won't build anything.
If you want to install AutoPPL into the system, remove `-DCMAKE_INSTALL_PREFIX=..`.
The `make install` will install the `include` directory and CMake shared files in `build` directory.

In your own CMake project, assuming you have a single source file 
`main.cpp` in the same directory as your CMakeLists.txt,
write the following as a minimal configuration:
```cmake
cmake_minimum_required(VERSION 3.7)
project("MyProject")
find_package(AutoPPL CONFIG REQUIRED)
find_package(FastAD CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
add_executable(main main.cpp)
target_link_libraries(main AutoPPL::AutoPPL FastAD::FastAD Eigen3::Eigen)
```

If you installed `AutoPPL` locally as instructed before,
you should add a hint to the `find_package` like:
```
find_package(AutoPPL CONFIG REQUIRED HINTS path_to_autoppl/build/share)
```

The above rule applies for `FastAD` and `Eigen3` as well.
If you installed these libraries using `setup.sh` locally,
the required hint paths are:
```
find_package(FastAD CONFIG REQUIRED HINTS path_to_autoppl/lib/FastAD/build/share)
find_package(Eigen3 CONFIG REQUIRED HINTS path_to_autoppl/lib/FastAD/libs/Eigen3/build/share)
```

## Quick Guide

We assume that we are in `namespace ppl` throughout this section.

### Variable

There are only a few variable types that a user will need to define a model.
```cpp
DataView<ValueType, ShapeType> Xv(ptr, rows, cols);
Data<ValueType, ShapeType> X(rows, cols);
Param<ValueType, ShapeType, ConstraintType> p(rows, cols);
TParam<ValueType, ShapeType> tp(rows, cols);
```

For all types listed above, `ValueType` should be either `double` or `int`, and
`ShapeType` must be one of `scl, vec, mat` (scalar, column vector, matrix) to indicate the general shape.
By default, `ShapeType` is `scl`.
Depending on the shape, the user may omit `rows` or `cols`.
For example, if `ShapeType` is `scl`, one can omit both `rows` and `cols` (both set to 1),
and if `ShapeType` is `vec`, one can omit `cols` (set to 1).
For `Param` specifically, see [Constraint](#constraint) for more information about `ConstraintType`.
For this section, this third template parameter is not important.

The difference between `DataView` and `Data` is that 
`DataView` only views existing data (does not copy any data) and `Data` _owns_ data.
`DataView` will view data in column major format and `Data` will own data in column major format.
To get the underlying data that `DataView` views or `Data` owns, 
we expose the member function `get`, which will return a reference to `Eigen::Map` or `Eigen::Matrix`, respectively.
We ask the users to refer to `Eigen` documentation for modifying the data.

`Param` and `TParam` objects do not own or view any values.
There is nothing to do from the user other than constructing them. 
More details on their distinction will become clearer in the later sections.
At a high level, a `Param` is a parameter that can be sampled
and a `TParam` is a transformation of parameters and is not sampled.

It is worth mentioning that currently `TParam<T, vec>` objects are the only ones
that provide subsetting, i.e. have `operator[]` defined.
There should be no need with `Data` or `DataView` since one can subset the underlying Eigen object itself
and create a new `Data` or `DataView` object to own a copy of or view that subset.
Note that `tp[i]` returns another expression and does not actually retrieve any value;
it is simply a lightweight object that refers to `tp` and the `i`th value it represents.
Users should not concern themselves how `tp` actually binds to values during MCMC.

### Variable Expression

Variable expressions are any expressions that are "mathematical" functions of variables.
We provide overloads for `operator+,-,*,/,+=,-=,*=,/=,=`, 
functions such as `sin, cos, tan, log, exp, sqrt, dot, for_each`.
All functions are vectorized whenever possible.
Here is an example:
```cpp
Data<double, mat> X(m,n);
Param<double, vec> w(n), w2(m);
TParam<double, scl> s;
auto var_expr = sin(dot(X, w) + s) - w2;
```

The expression above first creates an expression for the dot-product,
then a vectorized sum with a scalar (`s`),
then a vectorized `sin`,
then finally vectorized `operator-` with `w2`.

The only non-obvious function is `for_each`.
It has the same syntax as `std::for_each`,
however, the lambda function must return some variable expression:
```cpp
TParam<double, vec> h(10);
auto expr = for_each(util::counting_iterator<size_t>(0),
                     util::counting_iterator<size_t>(h.size()),
                     [&](size_t i) { return h[i] += h[i-1] * 2.; });
```

Note that we provide our own `counting_iterator` since they become quite useful in this context.
One can think of this expression as a "lazy-version" of the following:

```cpp
for (size_t i = 0; i < h.size(); ++i) {
    h[i] += h[i-1] * 2.;
}
```
assuming `h` in this context is, for example, `std::vector<double>`.
Again, nothing is computed when the expression is constructed.
All computation is done lazily during MCMC sampling.

Finally, users can create constants by writing literals directly, 
as shown above, when constructing variable expressions.
If the user wishes to create a constant vector or matrix, 
they can simply use those objects when constructing the variable expression
and our library will wrap them as constant objects.
The difference between constants and data is that constants cannot be assigned a distribution.

### Constraint

This section only applies to `Param` objects.
Sometimes, parameters must be constrained.
Some notable examples are 
covariance matrix (symmetric and positive definite), 
probability values (bounded by 0 and 1), 
and standard deviation (bounded below by 0).
It is a well-known problem that sampling constrained parameters directly is highly-inefficient.

Every parameter has an associated unconstrained and constrained value.
Most MCMC algorithms like NUTS and Metropolis-Hastings will sample unconstrained values,
and transform the unconstrained values to constrained values when computing the log-pdf (corrected with the Jacobian).
For more information on how these transformations are performed, we direct the readers to look at
[STAN reference guide](https://mc-stan.org/docs/2_21/reference-manual/variable-transforms-chapter.html).
Note that users will always receive _constrained_ values as their samples after invoking a MCMC sampler.

Currently we only support lower bounds, lower-and-upper bounds, (symmetric) positive-definite, and no constraint.
We recommend using C++17 class template argument deduction (CTAD) 
instead of `auto` to indicate that these objects are indeed parameters,
but of course, one can certainly just use `auto`:
```cpp
// Declaration pseudo-code:

// make_param<ValueType, ShapeType=scl, ConstraintType>(rows, cols, constraint);
// make_param<ValueType, ShapeType=scl, ConstraintType>(rows, constraint);
// make_param<ValueType, ShapeType=scl, ConstraintType>(constraint);

Param sigma = make_param<double>(lower(0.));                // scalar lower bounded by 0.
Param p = make_param<double, vec>(10, bounded(0., 1.));     // 10-element vector bounded by 0., 1.
Param Sigma = make_param<double, mat>(3, pos_def());        // 3x3 covariance matrix
```
From section [Variable](#variable), we saw that `Param` had a third template parameter.
Using the `make_param` helper function, we can deduce that third parameter type,
which is precisely the constraint argument to `make_param`.
In general, the constraint can be a complicated expression depending on other parameters as such:
```cpp
Param<double> w1;
Param w2 = make_param<double>(lower(w1 - 1. + w1 * w1));
```
so it is crucial that we are able to deduce the type.

### Distribution Expression

A distribution expression internally defines how to compute the log-pdf (dropping any constants).
Currently we support the following distributions:

| Distribution | Syntax |
| ------------ | ------ |
| [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution) | `ppl::bernoulli(p)` |
| [Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution) | `ppl::cauchy(x0, gamma)` |
| [Normal](https://en.wikipedia.org/wiki/Normal_distribution) | `ppl::normal(mu, sigma)` |
| [Multivariate Normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) | `ppl::normal(mu, Sigma)` |
| [Uniform](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)) | `ppl::uniform(min, max)` |
| [Wishart](https://en.wikipedia.org/wiki/Wishart_distribution) | `ppl::wishart(V, n)` |

Here are some examples:
```cpp
Param<double> m1, m2, M1, M2;
auto uniform_expr = uniform(m1 + m2, M1 + M2);

Param<double> l1, l2, s;
auto normal_expr = normal(l1 + l2, s);

Param V = make_param<double, mat>(3, pos_def);
auto wishart_expr = wishart(V, 5);
```

### Model Expression

There are two operators that govern model expressions: `operator|=` and `operator,`.
`operator|=` assigns a distribution to a parameter or data and 
`operator,` "glues" such assignments together, defining a joint distribution.
The choice for `operator|=` is deliberate because it captures both the intuition that we are
assigning a distribution to a variable and that the distribution is _conditional_ (`|`).

Example:
```cpp
Data<double, vec> y(10);
Param<double> m1, m2, M1, M2;
Param<double> l1, l2, s;

auto model = (
    m1 |= normal(1., 1.),
    m2 |= normal(-1., 1.),
    M1 |= uniform(m2, m1),
    M2 |= uniform(M, 3.24 + m1),
    y |= normal(l1 + l2, s)
);
```

### Transformed Parameters

There are cases where defining a model expression is not enough.
Sometimes parameters have to be further transformed to cache some common transformation.
For example, here is a stochastic volatility model taken from
[STAN](https://mc-stan.org/docs/2_21/stan-users-guide/stochastic-volatility-models.html),
but using AutoPPL:

```cpp
DataView<double, vec> y(ptr, n_data);
Param phi = make_param<double>(bounded(-1., 1.));
Param sigma = make_param<double>(lower(0.));
Param<double> mu;
Param<double, vec> h_std(n_data);
TParam<double, vec> h(n_data);

// define transformed parameter expression
auto tp_expr = (
    h = h_std * sigma,
    h[0] /= sqrt(1. - phi * phi),
    h += mu,
    for_each(util::counting_iterator<>(1),
             util::counting_iterator<>(h.size()),
             [&](size_t i) { return h[i] += phi * (h[i-1] - mu); })
);

auto model = (
    phi |= uniform(-1., 1.),
    sigma |= cauchy(0., 5.),
    mu |= cauchy(0., 10.),
    h_std |= normal(0., 1.),
    y |= normal(0., exp(h / 2.))
);
```

A couple of notes:
- transformed parameter expression _must_ be defined as a separate expression from model expression
- it is more efficient to make a transformed parameter expression 
  if there is a common transformation used in multiple places when defining a model expression
- all variable expressions are perfectly valid when defining TP expressions, 
  but note that `operator=` is only available for `TParam` objects.
  In fact, all `TParam` objects that are used in a model expression _must have exactly one_
  `operator=` expression assigning some expression to that `TParam` object. 
  The user cannot assign a vector `TParam` with some expression and then also assign a subview again like:
  ```cpp
  h = h_std * sigma,
  h[0] = 1.
  ```
  because this introduces ambiguity when back-evaluating during automatic differentiation.
  However, it is fine to assign each subview exactly once like:
  ```cpp
  for_each(util::counting_iterator<>(0),
           util::counting_iterator<>(h.size),
           [&](size_t i) { return h[i] = h_std[i] * sigma; });
  ```
- it is possible to assign a `TParam` with an expression composed of only data and constants,
  but this is much less efficient than precomputing that expression and creating a new constant out of that.
  This precomputation only requires using Eigen library and should not be a part of any autoppl expressions.

### Program Expression

A program expression simply combines a transformed parameter expression with a model expression.
If there is no transformed parameter expression, then a program expression simply wraps a model expression.
A program expression is what gets passed to MCMC samplers.
The user does not need to convert a model expression into a program expression, 
but if there is a transformed parameter expression,
the user should use `operator|` to "pipe" transformed parameter expression with a model expression
to create a program expression (order matters! The wrong order will raise a compiler error):

```cpp
auto program = tp_expr | model;
```

### Sampling Algorithms

Currently, we support the following algorithms:

| MCMC Algorithm | Syntax |
| -------------- | ------ |
| [Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) | ppl::mh(program, config) |
| [No-U-Turn Sampler (NUTS)](http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf) | ppl::nuts(program, config) |

Every sampling algorithm has a corresponding configuration object associated with it.
The user does not need to pass a configuration, in which case, a default-constructed object gets passed with the default settings.

We briefly list the configuration declaration:

```cpp
struct ConfigBase
{
    size_t warmup = 1000;               // number of warmup iterations
    size_t samples = 1000;              // number of samples
    size_t seed = mcmc::random_seed();  // seed (default: random)
    bool prune = true;                  // extra step during initialization to make sure log-pdf is not degenerate
};

// MH-specific

struct MHConfig : ConfigBase
{
    double sigma = 1.0;     // proposal distribution SD (N(0, sigma))
    double alpha = 0.25;    // discrete proposal distribution (triangular on {-1,0,1})
                            // with alpha probability on -1 and 1 (1-2*alpha on 0).
};

// NUTS-specific

struct StepConfig
{
    double delta = 0.8;
    double gamma = 0.05;
    double t0 = 10.;
    double kappa = 0.75;
};

struct VarConfig
{
    size_t init_buffer = 75;
    size_t term_buffer = 50;
    size_t window_base = 25;
};

// template parameter one of: unit_var, diag_var
template <class VarAdapterPolicy=diag_var>
struct NUTSConfig: ConfigBase
{
    using var_adapter_policy_t = VarAdapterPolicy;
    size_t max_depth = 10;
    StepConfig step_config;
    VarConfig var_config;
};
```

For the NUTS-specific configuration, we direct the reader to 
[STAN](https://mc-stan.org/docs/2_18/reference-manual/hmc-algorithm-parameters.html).

Every sampler will return a `ppl::MCMCResult<>` object.
The template parameter indicates the row or column-major for the underlying sample matrix.
The sampler may choose to sample the unconstrained values and write in a row-major result object
for speed purposes and then convert to a column-major result object with constrained values.
We briefly show the class declaration:

```cpp
template <int Major = Eigen::ColMajor>
struct MCMCResult
{
    // ...
    cont_samples_t cont_samples;    // (continuous) sample matrix
    disc_samples_t disc_samples;    // (discrete) sample matrix
    std::string name;               // MCMC algorithm name
    double warmup_time = 0;         // time elapsed for warmup
    double sampling_time = 0;       // time elapsed for sampling 
};
```

The sample matrices will always be `samples x n_constrained_values`,
where `samples` is the number of samples requested from the config object
and `n_constrained_values` is the number of constrained parameter values (flattened into a row).
The algorithms guarantee that each row consists of sampled parameter values 
in the same order as the priors in the model.
As an example, for the following model
```cpp
auto model = (
    t1 |= ...,
    t2 |= ...,
    t3 |= ...,
    ...
);
```
a sample row would consist of values for `t1`, `t2`, `t3` in that order.
If the parameters are multi-dimensional, their values are flattened assuming column-major format.
So if `t1` is a vector with 4 elements, the first four elements of a row will be values for `t1`.
If `t2` is a 2x2 matrix, the next four elements of a row will be values for `t2(0,0), t2(1,0), t2(0,1), t2(1,1)`.

Currently, we do not support a `summary` function yet to output a summary of the samples.
The user can, however, directly call `res.cont_samples.colwise().mean()` to compute the mean for each column.
We also provide `ppl::math::ess(matrix)` to compute column-wise effective-sample-size (ESS).
The user can then divide this result with `res.sampling_time` to get `ESS/s`,
which is the preferred metric to compare performance among MCMC algorithms.
ESS was computed as outlined 
[here](https://mc-stan.org/docs/2_23/reference-manual/effective-sample-size-section.html).
We also made some adjustments to use Geyer's biased estimator for ESS
as in the current implementation of STAN
([source](https://github.com/stan-dev/stan/blob/525998129ea838ec685f1d1f65dc76063d0fd40d/src/stan/analyze/mcmc/compute_effective_sample_size.hpp)).

## Examples

### Sampling from Joint Distribution

Although AutoPPL was designed to perform inference on posterior distributions,
one can certainly use it to sample from any joint distribution defined by the priors and conditional distributions.
For example, we can sample `1000` points with `1000` warmup iterations from a 
standard normal distribution using Metropolis-Hastings in the following way:

```cpp
Param<double> theta;
auto model = (theta |= normal(0., 1.));
auto res = mh(model, config);
```

In general, so long as the joint PDF is known, 
or equivalently and more commonly if the conditional and prior PDFs are known,
one can sample from the distribution.
As another example, we may sample from a more complicated joint distribution:
```cpp
Param<double> theta1;
Param<double> theta2;
auto model = (
    theta1 |= uniform(-1., 1.),
    theta2 |= normal(theta1, 1.)
);
auto res = mh(model); 
```

### Sampling Posterior Mean and Standard Deviation

The following is an example of fitting a Gaussian model to some data.
We put a `Normal(0,3)` prior on the mean and `Uniform(0,2)` prior on the 
standard deviation.
While in the previous section, we used Metropolis-Hastings 
to demonstrate how to use it,
it is recommended to use the state-of-the-art NUTS sampler to sample
from the posterior distribution.

```cpp
Data<double, vec> x(5);
Param<double> mu;
Param<double> sigma;

x.get() << 1.0, 1.5, 1.7, 1.2, 1.5; // load data

auto model = (
    mu |= normal(0., 3.),
    sigma |= uniform(0., 2.),
    x |= normal(mu, sigma)
);

auto res = nuts(model);
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
Data<double> x(6); 
Data<double> y(6);
Param<double> w;
Param<double> b;

x.get() << 2.4, 3.1, 3.6, 4, 4.5, 5.;
y.get() << 3.5, 4, 4.4, 5.01, 5.46, 6.1;

auto model = (
        w |= uniform(0., 2.),
        b |= uniform(0., 2.),
        y |= normal(x * w + b, 0.5)
);

auto res = nuts(model);
```

### Stochastic Volatility

This example was discussed in a previous section, but we mention it again as reference:
```cpp
DataView<double, vec> y(ptr, n_data);
Param phi = make_param<double>(bounded(-1., 1.));
Param sigma = make_param<double>(lower(0.));
Param<double> mu;
Param<double, vec> h_std(n_data);
TParam<double, vec> h(n_data);

auto tp_expr = (
    h = h_std * sigma,
    h[0] /= sqrt(1. - phi * phi),
    h += mu,
    for_each(util::counting_iterator<>(1),
             util::counting_iterator<>(h.size()),
             [&](size_t i) { return h[i] += phi * (h[i-1] - mu); })
);

auto model = (
    phi |= uniform(-1., 1.),
    sigma |= cauchy(0., 5.),
    mu |= cauchy(0., 10.),
    h_std |= normal(0., 1.),
    y |= normal(0., exp(h / 2.))
);

auto res = nuts(tp_expr | model);
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
in both sampling and warmup times by a factor of about 6.5-7.
As for ESS/s, upon comparing by colors (corresponding to a parameter)
between dotted (STAN) and solid (AutoPPL) lines,
we see that AutoPPL has uniformly larger ESS/s by a factor of 6.5-7 as well.
This difference quickly becomes more noticeable as sample size grows.
From these plots and that sampling results were identical
show that the drastic difference in ESS/s is simply from faster
automatic differentation and a good use of memory to take advantage of cache.

The following is the AutoPPL code for the model specification without data loading.
The full code can be found
[here](benchmark/regression_autoppl.cpp):

```cpp
DataView<double, mat> X(X_data.data(), X_data.rows(), X_data.cols());
DataView<double, vec> y(y_data.data(), y_data.rows());
Param<double, vec> w(3);
Param<double> b;
Param<double> s;

auto model = (s |= uniform(0.5, 8.),
              b |= normal(0., 5.),
              w |= normal(0., 5.),
              y |= normal(dot(X, w) + b, s * s + 2.));

NUTSConfig<> config;
config.warmup = num_samples;
config.samples = num_samples;

auto res = nuts(model, config);
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
Comparing the sampling times, for example, we see about 20 times improvement.
The ESS/s for `l1` and `l2` overlap completely (red and blue) in both STAN and AutoPPL
and this is expected since they are symmetric in the model specification.
With the exception of the two smallest sample sizes (100, 500),
ESS/s is fairly constant as sample size varies.
It is quite evident that AutoPPL (solid) has a larger ESS/s by a factor of 20.

The reason for this difference is in how we handle expressions
where data vector elements are iid (independent and identically distributed).
For most distributions, especially those that are in some exponential family,
they can be highly optimized in iid settings to perform quicker differentiation.
However, it is worth noting that this optimization does not apply when
the data are simply independent but not identically distributed
(as in the [linear regression](#benchmarks-bayesian-linear-regression) case),
or when the variable is a parameter, not data.
Nonetheless, our AD is extremely fast due to vectorization and is in general faster than STAN.

The following is the AutoPPL code without data generation.
The full code can be found 
[here](benchmark/normal_two_prior_distribution.cpp).

```cpp
Data<double, vec> y(n_data);
Param<double> lambda1, lambda2, sigma;

auto model = (
    sigma |= uniform(0.0, 20.0),
    lambda1 |= normal(0.0, 10.0),
    lambda2 |= normal(0.0, 10.0),
    y |= normal(lambda1 + lambda2, sigma)
);

NUTSConfig<> config;
config.n_samples = n_samples;
config.warmup = n_samples;

auto res = nuts(model, config);
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
- [Clang](https://clang.llvm.org/): one of the main compilers used.
- [CMake](https://cmake.org/): build system.
- [Coveralls](https://coveralls.io/): check test coverage.
- [Cpp Coveralls](https://github.com/eddyxu/cpp-coveralls): check test coverage specifically for C++ code.
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page): matrix library.
- [FastAD](https://github.com/JamesYang007/FastAD): automatic differentiation library.
- [GCC](https://gcc.gnu.org/): one of the main compilers used.
- [Google Benchmark](https://github.com/google/benchmark): benchmark library algorithms.
- [GoogleTest](https://github.com/google/googletest): unit/integration-tests.
- [Travis CI](https://travis-ci.org/): continuous integration for Linux using GCC.
- [Valgrind](http://valgrind.org/): check memory leak and errors.

