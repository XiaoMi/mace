FROM ubuntu:16.04

# Update source
# Looks like mirrors.163.com does not work in xiaomi network
# RUN sed -i 's/http:\/\/archive\.ubuntu\.com\/ubuntu\//http:\/\/mirrors\.163\.com\/ubuntu\//g' /etc/apt/sources.list
RUN apt-get update -y

## Basic tools
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    libcurl3-dev \
    libgoogle-glog-dev \
    libfreetype6-dev \
    libpng12-dev \
    libprotobuf-dev \
    libzmq3-dev \
    pkg-config \
    python-dev \
    python-pip \
    protobuf-compiler \
    rsync \
    software-properties-common \
    unzip \
    zip \
    zlib1g-dev \
    openjdk-8-jdk \
    openjdk-8-jre-headless \
    openssh-server \
    wget
RUN pip install --upgrade pip

ENV ANDROID_NDK_HOME /opt/android-ndk
ENV ANDROID_NDK /opt/android-ndk
ENV ANDROID_NDK_VERSION r12b

# Android NDK
# download
RUN mkdir /opt/android-ndk-tmp && \
    cd /opt/android-ndk-tmp && \
    wget -q https://dl.google.com/android/repository/android-ndk-${ANDROID_NDK_VERSION}-linux-x86_64.zip && \
# uncompress
    unzip -q android-ndk-${ANDROID_NDK_VERSION}-linux-x86_64.zip && \
# move to its final location
    mv ./android-ndk-${ANDROID_NDK_VERSION} ${ANDROID_NDK_HOME} && \
# remove temp dir
    cd ${ANDROID_NDK_HOME} && \
    rm -rf /opt/android-ndk-tmp

# add to PATH
ENV PATH ${PATH}:${ANDROID_NDK_HOME}

# Set up Bazel.
# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.7.0
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Setup vim
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends \
    locales \
    vim

RUN mkdir -p ~/.vim/autoload ~/.vim/bundle && \
    curl -LSso ~/.vim/autoload/pathogen.vim https://tpo.pe/pathogen.vim

RUN mkdir -p ~/.vim/bundle
RUN cd ~/.vim/bundle && \
    git clone https://github.com/scrooloose/nerdtree.git && \
    git clone https://github.com/vim-syntastic/syntastic.git && \
    git clone https://github.com/vim-airline/vim-airline.git && \
    git clone https://github.com/altercation/vim-colors-solarized.git && \
    git clone https://github.com/bazelbuild/vim-ft-bzl.git && \
    git clone https://github.com/google/vim-maktaba.git && \
    git clone https://github.com/google/vim-codefmt.git

RUN curl -LSso ~/.vimrc  https://gist.githubusercontent.com/llhe/71c5802919debd5825c100c0135478a7/raw/16a35020cc65f9c25d0cf8f11a3ba7b345a1271d/.vimrc

RUN locale-gen en_US.UTF-8
ENV LC_CTYPE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV TERM xterm-256color

# Extra packges
RUN apt-get install -y --no-install-recommends \
    telnet \
    net-tools \
    inetutils-ping \
    screen \
    android-tools-adb

# Install tools
RUN pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com setuptools
RUN pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com tensorflow==1.4.0 \
    scipy \
    jinja2 \
    pyyaml \
    sh 

# Download tensorflow tools
RUN wget http://cnbj1-inner-fds.api.xiaomi.net/mace/tool/transform_graph && \
    chmod +x transform_graph
