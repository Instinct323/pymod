#!/bin/bash

export TMP=/tmp/opencv
apt install -y libgtk2.0-dev libjpeg-dev libopenexr-dev libtbb-dev
git clone https://github.com/opencv/opencv $TMP
cmake-install.bash $TMP
cp -r /usr/local/include/opencv4/opencv2 /usr/local/include
