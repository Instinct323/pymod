# docker build -f cpp-ubuntu.dockerfile -t cpp-base .
# docker run -p 22:22 cpp-base
# docker exec -it <ctn> bash

FROM ubuntu:18.04

ARG USER=tongzj
ARG PASSWD='20010323'
ARG EMAIL='1400721986@qq.com'

RUN useradd -m $USER && \
    echo $USER:$PASSWD | chpasswd

# apt
RUN apt update && \
    apt install -y tree unzip wget

# Git
RUN apt install -y git && \
    git config --global user.name $USER && \
    git config --global user.email $EMAIL

# OpenSSH
RUN apt install -y openssh-server && \
    mkdir /var/run/sshd && \
    sed -ri 's/^PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

# C++ toolchain
ARG BIN=/usr/local/bin
COPY cpp-bin/*.bash $BIN/
RUN chmod +x $BIN/*.bash && \
    apt install -y build-essential cmake gdb

WORKDIR /home/$USER
CMD /usr/sbin/sshd -D
