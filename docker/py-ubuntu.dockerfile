# docker build -f py-ubuntu.dockerfile -t py .
# docker run -p 22:22 py
# docker exec -it <ctn> bash

FROM ubuntu:22.04

ARG USER=tongzj
ARG PASSWD='20010323'
ARG EMAIL='1400721986@qq.com'

# windows: VcXsrv
ENV DISPLAY='host.docker.internal:0'
ENV DEBIAN_FRONTEND=noninteractive

RUN useradd -m $USER && \
    echo $USER:$PASSWD | chpasswd && \
    echo root:$PASSWD | chpasswd && \
    usermod -aG sudo $USER

# apt
RUN apt update && \
    apt install -y tree unzip wget sudo

# Git
RUN apt install -y git && \
    git config --global user.name $USER && \
    git config --global user.email $EMAIL

# OpenSSH
RUN apt install -y openssh-server && \
    mkdir /var/run/sshd && \
    sed -ri 's/^PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config && \
    apt clean

# Miniconda
ARG CONDA=Miniconda3-py38_23.11.0-2-Linux-x86_64.sh
WORKDIR /tmp
RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/$CONDA && \
    chmod +x $CONDA && \
    ./$CONDA -b -p /opt/miniconda && \
    rm -f $CONDA
ENV PATH=/opt/miniconda/bin:$PATH

# Config: pip, conda
RUN pip config set global.timeout 6000 && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --set show_channel_urls yes

WORKDIR /home/$USER
CMD /usr/sbin/sshd -D
