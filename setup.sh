#!/bin/bash

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
