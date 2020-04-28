#!/bin/bash

projectdir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

gbenchpath="lib/benchmark"
gtestpath="$gbenchpath/googletest"

# setup google benchmark
git submodule update --init

# setup googletest
if [ ! -d "$gtestpath" ]; then
    git clone https://github.com/google/googletest.git $gtestpath
    cd $gtestpath 2>&1 /dev/null
    git checkout -q release-1.10.0
    cd ~- # change back to previous dir and no output to terminal
fi

# setup Armadillo
if [[ "$OSTYPE" == "linux-gnu" || "$OSTYPE" == "darwin"* ]]; then
    cd /tmp
    if [[ "$OSTYPE" == "linux-gnu" ]]; then
        sudo apt install libopenblas-dev liblapack-dev
    fi
    wget http://sourceforge.net/projects/arma/files/armadillo-9.870.2.tar.xz
    if [ -d "armadillo-9.870.2" ]; then
        rm -rf armadillo-9.870.2
    fi
    tar -xvf armadillo-9.870.2.tar.xz
    cd armadillo-9.870.2
    cmake . -DCMAKE_INSTALL_PREFIX="$projectdir/lib/armadillo"
    make
    make install
    cd $projectdir
fi

# setup FastAD
cd /tmp
if [ ! -d "FastAD" ]; then
    git clone https://github.com/JamesYang007/FastAD.git
fi
cd FastAD && git pull
./setup.sh
./clean-build.sh release -DFASTAD_ENABLE_TEST=OFF \
    -DCMAKE_INSTALL_PREFIX="$projectdir/lib/FastAD"
cd build/release
ninja install
cd $projectdir
