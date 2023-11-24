# docker build -f cpp-centos.dockerfile -t cpp-base .
# docker run -p 22:22 cpp-base
# docker exec -it <ctn> bash

FROM centos:7
MAINTAINER TongZJ
# InComplete

# yum, wget
RUN yum install -y wget && \
    wget http://mirrors.aliyun.com/repo/Centos-7.repo -P /etc/yum.repos.d

# Git
RUN yum install -y git && \
    git config --global user.name 'TongZJ' && \
    git config --global user.email '1400721986@qq.com'

# OpenSSH
RUN yum install -y openssh-server && \
    mkdir /var/run/sshd && \
    sed -ri 's/^PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

CMD /usr/sbin/sshd -D
