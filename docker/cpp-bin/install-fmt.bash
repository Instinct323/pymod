#!/bin/bash

export TMP=/tmp/fmt
git clone https://github.com/fmtlib/fmt $TMP
cmake-install.bash $TMP
