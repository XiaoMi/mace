Environment requirement
========================

MACE requires the following dependencies:

Required dependencies
---------------------

.. list-table::
    :header-rows: 1

    * - Software
      - Installation command
      - Tested version
    * - Python
      -
      - 2.7
    * - Bazel
      - `bazel installation guide <https://docs.bazel.build/versions/master/install.html>`__
      - 0.13.0
    * - CMake
      - apt-get install cmake
      - >= 3.11.3
    * - Jinja2
      - pip install -I jinja2==2.10
      - 2.10
    * - PyYaml
      - pip install -I pyyaml==3.12
      - 3.12.0
    * - sh
      - pip install -I sh==1.12.14
      - 1.12.14
    * - Numpy
      - pip install -I numpy==1.14.0
      - Required by model validation
    * - six
      - pip install -I six==1.11.0
      - Required for Python 2 and 3 compatibility (TODO)

Optional dependencies
---------------------

.. list-table::
    :header-rows: 1

    * - Software
      - Installation command
      - Remark
    * - Android NDK
      - `NDK installation guide <https://developer.android.com/ndk/guides/setup#install>`__
      - Required by Android build, r15b, r15c, r16b, r17b
    * - ADB
      - apt-get install android-tools-adb
      - Required by Android run, >= 1.0.32
    * - TensorFlow
      - pip install -I tensorflow==1.6.0
      - Required by TensorFlow model
    * - Docker
      - `docker installation guide <https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository>`__
      - Required by docker mode for Caffe model
    * - Scipy
      - pip install -I scipy==1.0.0
      - Required by model validation
    * - FileLock
      - pip install -I filelock==3.0.0
      - Required by run on Android

.. note::

    - For Android build, `ANDROID_NDK_HOME` must be confifigured by using ``export ANDROID_NDK_HOME=/path/to/ndk``
    - It will link ``libc++`` instead of ``gnustl`` if ``NDK version >= r17b`` and ``bazel version >= 0.13.0``, please refer to `NDK cpp-support <https://developer.android.com/ndk/guides/cpp-support>`__.
