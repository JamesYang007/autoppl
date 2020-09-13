#!/bin/bash

gbenchpath="lib/benchmark"
gtestpath="$gbenchpath/googletest"
armapath="lib/armadillo"
fastadpath="lib/FastAD"

# setup google benchmark
if [ ! -d "$gbenchpath" ]; then
    git clone https://github.com/google/benchmark.git $gbenchpath
    cd $gbenchpath &>/dev/null
    git checkout -q v1.5.0
    cd ~- # change back to previous dir and no output to terminal
fi

# setup googletest
if [ ! -d "$gtestpath" ]; then
    git clone https://github.com/google/googletest.git $gtestpath
    cd $gtestpath &> /dev/null
    git checkout -q release-1.10.0
    cd ~- # change back to previous dir and no output to terminal
fi

# setup FastAD
if [ ! -d "$fastadpath" ]; then
    git clone https://github.com/JamesYang007/FastAD.git $fastadpath
    cd $fastadpath
    git checkout tags/v3.2.1
    ./setup.sh
    ./clean-build.sh release -DFASTAD_ENABLE_TEST=OFF \
        -DCMAKE_INSTALL_PREFIX=".." # installs into build
    cd build/release
    ninja install
    cd ../../ # in lib/FastAD
    cd ../../ # in working directory
fi
