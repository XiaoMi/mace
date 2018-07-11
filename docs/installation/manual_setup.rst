Manual setup
=============

The setup steps are based on ``Ubuntu``, you can change the commands
correspondingly for other systems.
For the detailed installation dependencies, please refer to :doc:`env_requirement`.

Install Bazel
-------------

Recommend bazel with version larger than ``0.13.0`` (Refer to `Bazel documentation <https://docs.bazel.build/versions/master/install.html>`__).

.. code:: sh

    export BAZEL_VERSION=0.13.1
    mkdir /bazel && \
        cd /bazel && \
        wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
        chmod +x bazel-*.sh && \
        ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
        cd / && \
        rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

Install Android NDK
--------------------

The recommended Android NDK versions includes r15b, r15c and r16b (Refers to
`NDK installation guide <https://developer.android.com/ndk/guides/setup#install>`__).

.. code:: sh

    # Download NDK r15c
    cd /opt/ && \
        wget -q https://dl.google.com/android/repository/android-ndk-r15c-linux-x86_64.zip && \
        unzip -q android-ndk-r15c-linux-x86_64.zip && \
        rm -f android-ndk-r15c-linux-x86_64.zip

    export ANDROID_NDK_VERSION=r15c
    export ANDROID_NDK=/opt/android-ndk-${ANDROID_NDK_VERSION}
    export ANDROID_NDK_HOME=${ANDROID_NDK}

    # add to PATH
    export PATH=${PATH}:${ANDROID_NDK_HOME}

Install extra tools
--------------------

.. code:: sh

    apt-get install -y --no-install-recommends \
        cmake \
        android-tools-adb
    pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com setuptools
    pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com \
        "numpy>=1.14.0" \
        scipy \
        jinja2 \
        pyyaml \
        sh==1.12.14 \
        pycodestyle==2.4.0 \
        filelock

Install TensorFlow (Optional)
------------------------------

.. code:: sh

    pip install -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com tensorflow==1.6.0


Install Caffe (Optional)
-------------------------

Please follow the installation instruction of `Caffe <http://caffe.berkeleyvision.org/installation.html>`__.
