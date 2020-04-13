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
