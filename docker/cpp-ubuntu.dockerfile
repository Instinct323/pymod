# docker build -f cpp-ubuntu.dockerfile -t cpp-base .
# docker run -p 22:22 cpp-base
# docker exec -it <ctn> bash

FROM ubuntu:18.04
MAINTAINER TongZJ

RUN useradd -m tongzj && \
    echo 'tongzj:20010323' | chpasswd

# apt
RUN apt update && \
    apt install -y tree unzip wget

# Git
RUN apt install -y git && \
    git config --global user.name 'TongZJ' && \
    git config --global user.email '1400721986@qq.com'

# OpenSSH
RUN apt install -y openssh-server && \
    mkdir /var/run/sshd && \
    sed -ri 's/^PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

# C++ toolchain
ARG BIN=/usr/local/bin
COPY *.bash $BIN/
RUN chmod +x $BIN/*.bash && \
    apt install -y build-essential cmake gdb net-tools

WORKDIR /home/tongzj
CMD /usr/sbin/sshd -D
