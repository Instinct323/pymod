#!/bin/bash
# cmake-install.bash <repo-path>

cmake-build.bash $1
cd $1/build

make install
rm -rf $1
