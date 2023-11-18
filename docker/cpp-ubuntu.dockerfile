# docker build -f cpp-ubuntu.dockerfile -t cpp-base .
# docker run -p 22:22 cpp-base
# docker exec -it <ctn> bash

FROM ubuntu:18.04
MAINTAINER TongZJ

RUN useradd -m tongzj
RUN echo 'tongzj:20010323' | chpasswd

# apt-get, wget
RUN apt-get update
RUN apt-get install -y wget
WORKDIR /home

# C++ toolchain
RUN apt-get install -y build-essential
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y cmake
RUN apt-get install -y gdb
RUN apt-get install -y net-tools

# Git
RUN apt-get install -y git
RUN git config --global user.email '1400721986@qq.com'
RUN git config --global user.name 'TongZJ'

# OpenSSH
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN sed -ri 's/^PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
CMD /usr/sbin/sshd -D
