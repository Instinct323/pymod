#!/bin/bash
# cmake-install.bash <repo-path>

if [ $(id -u) -eq 0 ]; then
  cmake-build.bash $1
  cd $1/cmake-build

  make install
  rm -rf $1

else
  echo "error: permission denied."
fi
