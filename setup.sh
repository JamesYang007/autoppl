#!/bin/bash

# Update submodule if needed
git submodule update --remote

# Setup google benchmark and googletest
if [ -d "libs/benchmark/googletest" ]; then
    cd libs/benchmark/googletest
    git pull
    cd -
else
    git clone https://github.com/google/googletest.git libs/benchmark/googletest
fi

cd libs/benchmark
mkdir -p build && cd build
cmake ../
cmake --build . -- -j12
