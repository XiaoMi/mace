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
RUN pyenv install 2.7.14 && pyenv install 3.6.3
RUN pyenv global 2.7.14

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

# Download NDK
RUN cd /opt/ && \
    wget -q https://dl.google.com/android/repository/android-ndk-r15c-linux-x86_64.zip && \
    unzip -q android-ndk-r15c-linux-x86_64.zip && \
    rm -f android-ndk-r15c-linux-x86_64.zip

ENV ANDROID_NDK_VERSION r15c
ENV ANDROID_NDK_HOME /opt/android-ndk-${ANDROID_NDK_VERSION}

# add to PATH
ENV PATH ${PATH}:${ANDROID_NDK_HOME}

# Install tools
RUN apt-get install -y --no-install-recommends \
    android-tools-adb
RUN pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com setuptools
RUN pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com \
    "numpy>=1.14.0" \
    scipy \
    jinja2 \
    pyyaml \
    sh==1.12.14 \
    pycodestyle==2.4.0 \
    filelock \
    PTable

# Install Tensorflow
RUN pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com \
    tensorflow==1.8.0

# Install pytorch (refer to: https://pytorch.org/get-started/locally/)
RUN pip install future==0.17.1 torchvision_nightly
RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

# Install ONNX
RUN pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com \
    onnx==1.3.0 \
    onnx-tf==1.2.0

# fix docker in docker error: `error while loading shared libraries: libltdl.so.7`
# refer to: https://github.com/jenkinsci/docker/issues/506
RUN apt-get install -y libltdl7
