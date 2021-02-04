FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    zip \
    wget \
    g++-5 \
    python \
    python-pip \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4-Linux-x86_64.sh && \
    chmod +x cmake-3.18.4-Linux-x86_64.sh && \
    ./cmake-3.18.4-Linux-x86_64.sh --skip-license --prefix=/usr && \
    rm cmake-3.18.4-Linux-x86_64.sh

RUN python -m pip install -U pip && pip install --no-cache-dir \
    cpplint pycodestyle\
    jinja2 pyyaml sh numpy six filelock \
    tensorflow==1.8.0 \
    sphinx sphinx-autobuild sphinx_rtd_theme recommonmark

RUN cd /opt/ && \
    wget -q https://dl.google.com/android/repository/android-ndk-r17b-linux-x86_64.zip && \
    unzip -q android-ndk-r17b-linux-x86_64.zip && \
    rm -f android-ndk-r17b-linux-x86_64.zip

RUN apt-get update && apt-get install -y android-tools-adb
RUN apt-get install -y git

RUN apt-get install -y gcc-arm-none-eabi
RUN apt-get install -y g++-arm-linux-gnueabihf


# RUN apt-get install -y python3 python3-pip
# RUN apt-get install -y mercurial

# RUN python3 -m pip install -U pip
# RUN python3 -m pip install jinja2 pyyaml sh numpy six filelock
# RUN python3 -m pip install tensorflow==2.3.0 tensorflow_model_optimization
# RUN python3 -m pip install mbed-cli