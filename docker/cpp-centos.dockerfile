# docker build -f cpp-centos.dockerfile -t cpp-base .
# docker run -p 22:22 cpp-base
# docker exec -it <ctn> bash

FROM centos:7
MAINTAINER TongZJ
# RUN echo 'root:20010323' | chpasswd

# yum, wget
RUN yum install -y wget
WORKDIR /etc/yum.repos.d
RUN wget http://mirrors.aliyun.com/repo/Centos-7.repo
WORKDIR /home

# C++ toolchain
RUN yum install -y deltarpm
RUN yum install -y cmake gcc gcc-c++ make gdb gdb-gdbserver

# Git
RUN yum install -y git
RUN git config --global user.email '1400721986@qq.com'
RUN git config --global user.name 'TongZJ'

# OpenSSH
RUN yum install -y openssh-server
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN ssh-keygen -A
CMD /usr/sbin/sshd -D
