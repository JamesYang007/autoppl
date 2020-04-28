#!/bin/bash

projectdir=$(dirname "BASH_SOURCE")
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

# install Armadillo only on Linux
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    sudo apt install libopenblas-dev liblapack-dev
    cd /tmp
    wget http://sourceforge.net/projects/arma/files/armadillo-9.870.2.tar.xz
    tar -xvf armadillo-9.870.2.tar.xz
    cd armadillo-9.870.2
    cmake .
    make
    sudo make install
    cd $projectdir
fi

# install FastAD
cd /tmp
git clone --recurse-submodules https://github.com/JamesYang007/FastAD.git FastAD
cd FastAD
./setup.sh
./install.sh
cd $projectdir
