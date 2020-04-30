#!/bin/bash

gbenchpath="lib/benchmark"
gtestpath="$gbenchpath/googletest"
armapath="lib/armadillo"
fastadpath="lib/FastAD"

# setup google benchmark
if [ ! -d "$gbenchpath" ]; then
    git clone https://github.com/google/benchmark.git $gbenchpath
    cd $gbenchpath 2>&1 /dev/null
    git checkout -q v1.5.0
    cd ~- # change back to previous dir and no output to terminal
fi

# setup googletest
if [ ! -d "$gtestpath" ]; then
    git clone https://github.com/google/googletest.git $gtestpath
    cd $gtestpath 2>&1 /dev/null
    git checkout -q release-1.10.0
    cd ~- # change back to previous dir and no output to terminal
fi

# setup Armadillo
if [ ! -d "$armapath" ]; then
    if [[ "$OSTYPE" == "linux-gnu" || "$OSTYPE" == "darwin"* ]]; then
        cd lib 2>&1 /dev/null
        if [[ "$OSTYPE" == "linux-gnu" ]]; then
            sudo apt install libopenblas-dev liblapack-dev
        fi
        wget http://sourceforge.net/projects/arma/files/armadillo-9.870.2.tar.xz
        tar -xvf armadillo-9.870.2.tar.xz
        cd armadillo-9.870.2
        cmake . -DCMAKE_INSTALL_PREFIX="../armadillo"
        make
        make install
        cd ../../ 2>&1 /dev/null
    fi
fi

# setup FastAD
if [ ! -d "$fastadpath" ]; then
    git clone https://github.com/JamesYang007/FastAD.git $fastadpath
    cd $fastadpath
    ./setup.sh
    ./clean-build.sh release -DFASTAD_ENABLE_TEST=OFF \
        -DCMAKE_INSTALL_PREFIX=".." # installs into build
    cd build/release
    ninja install
    cd ../../ # in lib/FastAD
    cd ../../ # in working directory
fi
