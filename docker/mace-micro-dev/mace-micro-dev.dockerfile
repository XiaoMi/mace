FROM ubuntu:18.04

RUN apt-get update --fix-missing
RUN apt-get install -y wget

RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4-Linux-x86_64.sh && chmod +x cmake-3.18.4-Linux-x86_64.sh && ./cmake-3.18.4-Linux-x86_64.sh --skip-license --prefix=/usr

RUN apt-get install -y g++-5 gcc-5
RUN apt-get install -y gcc-arm-none-eabi
RUN apt-get install -y git mercurial
RUN apt-get install -y python python-pip
RUN apt-get install -y python3 python3-pip

RUN python -m pip install -U pip
RUN python -m pip install jinja2 pyyaml sh numpy six filelock
RUN python -m pip install tensorflow==1.8.0

RUN python3 -m pip install -U pip
RUN python3 -m pip install jinja2 pyyaml sh numpy six filelock
RUN python3 -m pip install tensorflow==2.3.0 tensorflow_model_optimization
RUN python3 -m pip install mbed-cli
