FROM registry.cn-hangzhou.aliyuncs.com/xiaomimace/mace-dev-lite:latest

# Install tools
RUN apt-get install -y --no-install-recommends \
    telnet \
    net-tools \
    inetutils-ping \
    screen \
    htop
RUN pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com \
    sphinx \
    sphinx-autobuild \
    sphinx_rtd_theme \
    recommonmark

# Customize vim
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

# Upgrade CMake
RUN wget https://cmake.org/files/v3.11/cmake-3.11.3-Linux-x86_64.tar.gz -P /tmp/ && \
    tar zxf /tmp/cmake-3.11.3-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local/ && \
    update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force && \
    rm -f /tmp/cmake-3.11.3-Linux-x86_64.tar.gz

# mace-dev-lite image already included NDK r15c
## Download other NDK r12b
RUN cd /opt/ && \
    wget -q https://dl.google.com/android/repository/android-ndk-r12b-linux-x86_64.zip && \
    unzip -q android-ndk-r12b-linux-x86_64.zip && \
    rm -f android-ndk-r12b-linux-x86_64.zip
## Download other NDK r16
RUN cd /opt/ && \
    wget -q https://dl.google.com/android/repository/android-ndk-r16-linux-x86_64.zip && \
    unzip -q android-ndk-r16-linux-x86_64.zip && \
    rm -f android-ndk-r16-linux-x86_64.zip
## Download other NDK r17b
RUN cd /opt/ && \
    wget -q https://dl.google.com/android/repository/android-ndk-r17b-linux-x86_64.zip && \
    unzip -q android-ndk-r17b-linux-x86_64.zip && \
    rm -f android-ndk-r17b-linux-x86_64.zip
