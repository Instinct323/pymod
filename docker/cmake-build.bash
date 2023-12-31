#!/bin/bash
# cmake-build.bash <repo-path>

mkdir $1/cmake-build
cd $1/cmake-build

cmake ..
make
