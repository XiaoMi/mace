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
      - 2.7 or 3.6
    * - Bazel
      - `bazel installation guide <https://docs.bazel.build/versions/master/install.html>`__
      - 0.13.0
    * - CMake
      - Linux:``apt-get install cmake`` Mac:``brew install cmake``
      - >= 3.11.3
    * - Jinja2
      - pip install jinja2==2.10
      - 2.10
    * - PyYaml
      - pip install pyyaml==3.12
      - 3.12.0
    * - sh
      - pip install sh==1.12.14
      - 1.12.14
    * - Numpy
      - pip install numpy==1.14.0
      - Required by model validation
    * - six
      - pip install six==1.11.0
      - Required for Python 2 and 3 compatibility

For Bazel, install it following installation guide. For python dependencies,

	.. code:: sh

		pip install -U --user setup/requirements.txt



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
    * - CMake
      - apt-get install cmake
      - >= 3.11.3
    * - ADB
      - Linux:``apt-get install android-tools-adb`` Mac:``brew cask install android-platform-tools``
      - Required by Android run, >= 1.0.32
    * - TensorFlow
      - pip install tensorflow==1.8.0
      - Required by TensorFlow model
    * - Docker
      - `docker installation guide <https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository>`__
      - Required by docker mode for Caffe model
    * - Scipy
      - pip install scipy==1.0.0
      - Required by model validation
    * - FileLock
      - pip install filelock==3.0.0
      - Required by run on Android
    * - ONNX
      - pip install onnx==1.3.0
      - Required by ONNX model

For python dependencies,

	.. code:: sh

		pip install -U --user setup/optionals.txt


.. note::

    - For Android build, `ANDROID_NDK_HOME` must be confifigured by using ``export ANDROID_NDK_HOME=/path/to/ndk``
    - It will link ``libc++`` instead of ``gnustl`` if ``NDK version >= r17b`` and ``bazel version >= 0.13.0``, please refer to `NDK cpp-support <https://developer.android.com/ndk/guides/cpp-support>`__.
    - For Mac, please install Homebrew at first before installing other dependencies. Set ANDROID_NDK_HOME in ``/etc/bashrc`` and then run ``source /etc/bashrc``.  This installation was tested with macOS Mojave(10.14).
    
