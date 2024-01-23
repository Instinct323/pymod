#!/bin/bash

export TMP=/tmp/dbow3
git clone https://github.com/rmsalinas/DBow3 $TMP
cmake-install.bash $TMP
