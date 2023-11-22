#!/bin/bash
# cmake-install <repo-path>

cmake-build $1
cd $1/build

make install
rm -rf $1
