Environment Requirement
=======================

MACE requires the following dependencies:

Necessary Dependencies:
-----------------------

.. list-table::
    :header-rows: 1

    * - software
      - version
      - install command
    * - bazel
      - >= 0.13.0
      - `bazel installation guide <https://docs.bazel.build/versions/master/install.html>`__
    * - android-ndk
      - r15c/r16b
      - `NDK installation guide <https://developer.android.com/ndk/guides/setup#install>`__ or refers to the docker file
    * - adb
      - >= 1.0.32
      - apt-get install android-tools-adb
    * - cmake
      - >= 3.11.3
      - apt-get install cmake
    * - numpy
      - >= 1.14.0
      - pip install -I numpy==1.14.0
    * - scipy
      - >= 1.0.0
      - pip install -I scipy==1.0.0
    * - jinja2
      - >= 2.10
      - pip install -I jinja2==2.10
    * - PyYaml
      - >= 3.12.0
      - pip install -I pyyaml==3.12
    * - sh
      - >= 1.12.14
      - pip install -I sh==1.12.14
    * - filelock
      - >= 3.0.0
      - pip install -I filelock==3.0.0
    * - docker (for caffe)
      - >= 17.09.0-ce
      - `docker installation guide <https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository>`__

.. note::

    ``export ANDROID_NDK_HOME=/path/to/ndk`` to specify ANDROID_NDK_HOME

Optional Dependencies:
---------------------

.. list-table::
    :header-rows: 1

    * - software
      - version
      - install command
    * - tensorflow
      - >= 1.6.0
      - pip install -I tensorflow==1.6.0 (if you use tensorflow model)
