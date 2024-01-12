#!/bin/bash

export TMP=/tmp/Pangolin
apt install -y libglew-dev libboost-dev libboost-thread-dev libboost-filesystem-dev
git clone https://github.com/stevenlovegrove/Pangolin $TMP
cmake-install.bash $TMP
