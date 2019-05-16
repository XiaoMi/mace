FROM registry.cn-hangzhou.aliyuncs.com/xiaomimace/mace-dev-lite:latest

# Install tools
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends \
    telnet \
    net-tools \
    inetutils-ping \
    screen \
    htop

RUN pyenv global 3.6.3

RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    sphinx \
    sphinx-autobuild \
    sphinx_rtd_theme \
    recommonmark

# Customize vim
RUN git clone --recursive https://github.com/llhe/vimrc.git && \
    cd vimrc && \
    rm -rf ~/.vim && \
    rm -rf ~/.vimrc && \
    ln -s `pwd`/vim ~/.vim && \
    ln -s `pwd`/vimrc ~/.vimrc


# Upgrade CMake
RUN wget -q https://cmake.org/files/v3.11/cmake-3.11.3-Linux-x86_64.tar.gz -P /tmp/ && \
    tar zxf /tmp/cmake-3.11.3-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local/ && \
    update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force && \
    rm -f /tmp/cmake-3.11.3-Linux-x86_64.tar.gz

# mace-dev-lite image already included NDK r19c
## Download other NDK r15c
RUN cd /opt/ && \
    wget -q https://dl.google.com/android/repository/android-ndk-r15c-linux-x86_64.zip && \
    unzip -q android-ndk-r15c-linux-x86_64.zip && \
    rm -f android-ndk-r15c-linux-x86_64.zip

## Download other NDK r16b
RUN cd /opt/ && \
    wget -q https://dl.google.com/android/repository/android-ndk-r16b-linux-x86_64.zip && \
    unzip -q android-ndk-r16b-linux-x86_64.zip && \
    rm -f android-ndk-r16b-linux-x86_64.zip

## Download other NDK r17b
RUN cd /opt/ && \
    wget -q https://dl.google.com/android/repository/android-ndk-r17b-linux-x86_64.zip && \
    unzip -q android-ndk-r17b-linux-x86_64.zip && \
    rm -f android-ndk-r17b-linux-x86_64.zip

## Download other NDK r18b
RUN cd /opt/ && \
    wget -q https://dl.google.com/android/repository/android-ndk-r18b-linux-x86_64.zip && \
    unzip -q android-ndk-r18b-linux-x86_64.zip && \
    rm -f android-ndk-r18b-linux-x86_64.zip

# Mirror of https://releases.linaro.org/components/toolchain/binaries/7.3-2018.05/arm-linux-gnueabihf/gcc-linaro-7.3.1-2018.05-x86_64_arm-linux-gnueabihf.tar.xz
RUN cd /opt && \
    wget -q https://cnbj1.fds.api.xiaomi.com/mace/third-party/gcc-linaro/gcc-linaro-7.3.1-2018.05-x86_64_arm-linux-gnueabihf.tar.xz && \
    tar xf gcc-linaro-7.3.1-2018.05-x86_64_arm-linux-gnueabihf.tar.xz && \
    rm -rf gcc-linaro-7.3.1-2018.05-x86_64_arm-linux-gnueabihf.tar.xz

# Mirror of https://releases.linaro.org/components/toolchain/binaries/7.3-2018.05/aarch64-linux-gnu/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz
RUN cd /opt && \
    wget -q https://cnbj1.fds.api.xiaomi.com/mace/third-party/gcc-linaro/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz && \
    tar xf gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz && \
    rm -rf gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz

# install protoc
RUN cd opt/ && \
    wget -q https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip && \
    unzip protoc-3.6.1-linux-x86_64.zip -d protoc-3.6.1 && \
    rm -rf protoc-3.6.1-linux-x86_64.zip

ENV CROSS_TOOLCHAIN_PARENT /opt
ENV ANDROID_NDK_VERSION r19c
ENV ANDROID_NDK_HOME /opt/android-ndk-${ANDROID_NDK_VERSION}
ENV LINARO_ARM_LINUX_GNUEABIHF /opt/gcc-linaro-7.3.1-2018.05-x86_64_arm-linux-gnueabihf
ENV LINARO_AARCH64_LINUX_GNU /opt/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu
ENV PATH /opt/protoc-3.6.1/bin:${PATH}

RUN pyenv install 2.7.12
RUN pyenv global 2.7.12
RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --upgrade pip setuptools
RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    numpy==1.15.4 \
    scipy==1.2.0 \
    Jinja2==2.10 \
    PyYAML==3.13 \
    sh==1.12.14 \
    pycodestyle==2.4.0 \
    filelock==3.0.10 \
    PTable==0.9.2

RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    tensorflow==1.8.0

# Install pytorch (refer to: https://pytorch.org/get-started/locally/)
RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    future==0.17.1 \
    Pillow==5.4.1 \
    torch==1.1.0 \
    torchvision==0.2.2.post3

RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    onnx==1.3.0 \
    onnx-tf==1.2.0

RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    cpplint==1.4.4
