FROM ubuntu:16.04

WORKDIR /

# Update source
RUN apt-get update -y

# Basic tools
RUN apt-get install -y --no-install-recommends \
    apt-utils \
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
    protobuf-compiler \
    libprotoc-dev \
    rsync \
    software-properties-common \
    unzip \
    zip \
    zlib1g-dev \
    openjdk-8-jdk \
    openjdk-8-jre-headless \
    openssh-server \
    wget \
    bsdmainutils

# install pyenv
RUN apt-get install -y make \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python-openssl
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
ENV PATH /root/.pyenv/bin:/root/.pyenv/shims/:${PATH}
RUN eval "$(pyenv init -)"
RUN eval "$(pyenv virtualenv-init -)"
RUN pyenv install 3.6.3
RUN pyenv global 3.6.3

# Setup vim
RUN apt-get install -y --no-install-recommends \
    locales \
    vim

RUN locale-gen en_US.UTF-8
ENV LC_CTYPE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV TERM xterm-256color

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
ENV BAZEL_VERSION 0.16.0
RUN mkdir /tmp/bazel && \
    cd /tmp/bazel && \
    wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -rf /tmp/bazel

# Install SDK
ENV ANDROID_SDK_VERSION 4333796
ENV ANDROID_BUILD_TOOLS_VERSION 26.0.2
ENV ANDROID_SDK_FILENAME sdk-tools-linux-${ANDROID_SDK_VERSION}.zip
ENV ANDROID_SDK_URL https://dl.google.com/android/repository/${ANDROID_SDK_FILENAME}
ENV ANDROID_API_LEVELS android-26
ENV ANDROID_HOME /opt/sdk
ENV PATH ${PATH}:${ANDROID_HOME}/tools:${ANDROID_HOME}/platform-tools
RUN mkdir -p /opt/sdk && \
    cd /opt && \
    wget -q ${ANDROID_SDK_URL} && \
    unzip ${ANDROID_SDK_FILENAME} -d ${ANDROID_HOME} && \
    rm ${ANDROID_SDK_FILENAME} && \
    yes | android update sdk --no-ui -a --filter tools,platform-tools,${ANDROID_API_LEVELS},build-tools-${ANDROID_BUILD_TOOLS_VERSION}
RUN ${ANDROID_HOME}/tools/bin/sdkmanager "cmake;3.6.4111459"

# Download NDK 19c
RUN cd /opt/ && \
    wget -q https://dl.google.com/android/repository/android-ndk-r19c-linux-x86_64.zip && \
    unzip -q android-ndk-r19c-linux-x86_64.zip && \
    rm -f android-ndk-r19c-linux-x86_64.zip

ENV ANDROID_NDK_VERSION r19c
ENV ANDROID_NDK_HOME /opt/android-ndk-${ANDROID_NDK_VERSION}

# Install tools
RUN apt-get install -y --no-install-recommends \
    android-tools-adb
# fix docker in docker error: `error while loading shared libraries: libltdl.so.7`
# refer to: https://github.com/jenkinsci/docker/issues/506
RUN apt-get install -y libltdl7

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
