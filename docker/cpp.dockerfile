# docker build -f cpp.dockerfile -t cpp .
# docker run -p 22:22 cpp
# docker exec -it <ctn> bash

FROM cpp-base
MAINTAINER TongZJ

ENV INCLUDE=/usr/include \
    INCLUDE_L=/usr/local/include

# Eigen
RUN apt install -y libeigen3-dev && \
    cp -r $INCLUDE/eigen3/Eigen $INCLUDE

# Ceres
RUN export TMP=/tmp/ceres && \
    apt install -y libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev && \
    git clone -b 2.1.0 https://github.com/ceres-solver/ceres-solver $TMP && \
    cmake-install.bash $TMP

# g2o
RUN export TMP=/tmp/g2o && \
    apt install -y libspdlog-dev libsuitesparse-dev qtdeclarative5-dev qt5-qmake libqglviewer-dev-qt5 && \
    git clone -b 20201223_git https://github.com/RainerKuemmerle/g2o $TMP && \
    cmake-install.bash $TMP

# Pangolin
RUN export TMP=/tmp/Pangolin && \
    apt install -y libglew-dev libboost-dev libboost-thread-dev libboost-filesystem-dev && \
    git clone https://github.com/stevenlovegrove/Pangolin $TMP && \
    cmake-install.bash $TMP

# OpenCV
RUN export TMP=/tmp/opencv && \
    apt install -y libgtk2.0-dev libjpeg-dev libopenexr-dev libtbb-dev && \
    git clone https://github.com/opencv/opencv $TMP && \
    cmake-install.bash $TMP && \
    cp -r $INCLUDE_L/opencv4/opencv2 $INCLUDE_L

# fmt
RUN export TMP=/tmp/fmt && \
    git clone https://github.com/fmtlib/fmt $TMP && \
    cmake-install.bash $TMP

# Sophus
RUN export TMP=/tmp/Sophus && \
    git clone https://github.com/strasdat/Sophus $TMP && \
    cmake-install.bash $TMP
