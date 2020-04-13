#!/bin/bash

gbenchpath="lib/benchmark"
gtestpath="$gbenchpath/googletest"

# Update submodule if needed
git submodule update --recursive --remote

# Setup google benchmark and googletest
if [ -d "$gtestpath" ]; then
    cd $gtestpath && git pull && cd - 2&>1 /dev/null
else
    git clone https://github.com/google/googletest.git $gtestpath
fi
