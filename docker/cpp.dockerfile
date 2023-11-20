# docker build -f cpp.dockerfile -t cpp .
# docker run -p 22:22 cpp
# docker exec -it <ctn> bash

FROM cpp-base
MAINTAINER TongZJ

ENV INCLUDE=/usr/include
ENV INCLUDE_L=/usr/local/include

# Eigen
RUN apt-get install -y libeigen3-dev
RUN cp -r $INCLUDE/eigen3/Eigen $INCLUDE

# Ceres
ARG TMP=/tmp/ceres
RUN apt-get install -y libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev
RUN git clone -b 2.1.0 https://github.com/ceres-solver/ceres-solver $TMP
RUN mkdir $TMP/build; cd $TMP/build; cmake ..; make; make install

# g2o
ARG TMP=/tmp/g2o
RUN apt-get install -y libspdlog-dev libsuitesparse-dev qtdeclarative5-dev qt5-qmake libqglviewer-dev-qt5
RUN git clone -b 20201223_git https://github.com/RainerKuemmerle/g2o $TMP
RUN mkdir $TMP/build; cd $TMP/build; cmake ..; make; make install

# OpenCV
ARG TMP=/tmp/opencv
RUN apt-get install -y libgtk2.0-dev libjpeg-dev libopenexr-dev libtbb-dev
RUN git clone https://github.com/opencv/opencv $TMP
RUN mkdir $TMP/build; cd $TMP/build; cmake ..; make; make install
RUN cp -r $INCLUDE_L/opencv4/opencv2 $INCLUDE_L

# fmt
ARG TMP=/tmp/fmt
RUN git clone https://github.com/fmtlib/fmt $TMP
RUN mkdir $TMP/build; cd $TMP/build; cmake ..; make; make install

# Sophus
ARG TMP=/tmp/Sophus
RUN git clone https://github.com/strasdat/Sophus $TMP
RUN mkdir $TMP/build; cd $TMP/build; cmake ..; make; make install

WORKDIR /home/tongzj
