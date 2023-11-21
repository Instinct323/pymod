#!/bin/bash
# cpp-install <git-repo>

mkdir $1/build
cd $1/build

cmake ..
make
make install
rm -rf $1
