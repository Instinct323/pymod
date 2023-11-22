#!/bin/bash
# cmake-build <repo-path>

mkdir $1/build
cd $1/build

cmake ..
make
