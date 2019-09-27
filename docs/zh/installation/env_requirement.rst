安装环境
========================

MACE 需要安装下列依赖：

必须依赖
---------------------

.. list-table::
    :header-rows: 1

    * - 软件
      - 安装命令
      - Python 版本
    * - Python
      -
      - 2.7 or 3.6
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
      - 仅测试使用
    

可选依赖
---------------------

.. list-table::
    :header-rows: 1

    * - 软件
      - 安装命令
      - 版本和说明
    * - Android NDK
      - `NDK 安装指南 <https://developer.android.com/ndk/guides/setup#install>`__
      - Required by Android build, r15b, r15c, r16b, r17b
    * - CMake
      - apt-get install cmake
      - >= 3.11.3
    * - ADB
      - Linux:``apt-get install android-tools-adb`` Mac:``brew cask install android-platform-tools``
      - Android 运行需要, >= 1.0.32
    * - TensorFlow
      - pip install tensorflow==1.8.0
      - Tensorflow 模型转换需要
    * - Docker
      - `docker 安装指南 <https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository>`__
      - docker 模式用户需要
    * - Scipy
      - pip install scipy==1.0.0
      - 模型测试需要
    * - FileLock
      - pip install filelock==3.0.0
      - Android 运行需要
    * - ONNX
      - pip install onnx==1.5.0
      - ONNX 模型需要

对于 Python 依赖，可直接执行,

	.. code:: sh

		pip install -U --user -r setup/optionals.txt


.. note::

    - 对于安卓开发, 环境变量  `ANDROID_NDK_HOME` 需要指定 ``export ANDROID_NDK_HOME=/path/to/ndk``    
    - Mac 用户请先安装 Homebrew. 在 ``/etc/bashrc`` 中设置 `ANDROID_NDK_HOME`，之后执行 ``source /etc/bashrc``.
    
