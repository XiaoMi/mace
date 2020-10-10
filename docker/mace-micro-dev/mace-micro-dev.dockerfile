FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y g++ gcc
RUN apt-get install -y gcc-arm-none-eabi
RUN apt-get install -y python3 python3-pip git mercurial

RUN wget https://cdn.cnbj1.fds.api.mi-img.com/mace/third-party/cmake-3.18.3-Linux-x86_64.sh
RUN chmod +x cmake-3.18.3-Linux-x86_64.sh && ./cmake-3.18.3-Linux-x86_64.sh --skip-license --prefix=/usr

RUN python3 -m pip install -U pip
RUN python3 -m pip install jinja2 pyyaml sh numpy six filelock
RUN python3 -m pip install tensorflow==2.3.0 tensorflow_model_optimization
RUN python3 -m pip install mbed-cli
