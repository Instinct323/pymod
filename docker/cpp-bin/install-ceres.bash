#!/bin/bash

export TMP=/tmp/ceres
apt install -y libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev
git clone -b 2.1.0 https://github.com/ceres-solver/ceres-solver $TMP
cmake-install.bash $TMP
