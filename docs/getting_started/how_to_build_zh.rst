使用介绍
============

模型格式支持
-------------

.. list-table::
    :widths: auto
    :header-rows: 1
    :align: left

    * - Platform
      - Explanation
    * - Tensorflow
      - >= 1.6.0. (first choice, convenient for Android NN API in the future)
    * - Caffe
      - >= 1.0.

环境要求
---------

``mace``\ 提供了包含开发运行所需环境的docker镜像，镜像文件可以参考\ ``./docker/``\ 。启动命令：

.. code:: sh

    sudo docker pull cr.d.xiaomi.net/mace/mace-dev
    sudo docker run -it --rm --privileged -v /dev/bus/usb:/dev/bus/usb --net=host -v /local/path:/container/path cr.d.xiaomi.net/mace/mace-dev /bin/bash

如果用户希望配置开发机上的环境，可以参考如下环境要求：

.. list-table::
    :widths: auto
    :header-rows: 1
    :align: left

    * - software
      - version
      - install command
    * - bazel
      - >= 0.13.0
      - `bazel installation <https://docs.bazel.build/versions/master/install.html>`__
    * - android-ndk
      - r15c/r16b
      - reference the docker file
    * - adb
      - >= 1.0.32
      - apt-get install android-tools-adb
    * - tensorflow
      - >= 1.6.0
      - pip install -I tensorflow==1.6.0 (if you use tensorflow model)
    * - numpy
      - >= 1.14.0
      - pip install -I numpy=1.14.0
    * - scipy
      - >= 1.0.0
      - pip install -I scipy=1.0.0
    * - jinja2
      - >= 2.10
      - pip install -I jinja2=2.10
    * - PyYaml
      - >= 3.12.0
      - pip install -I pyyaml=3.12
    * - sh
      - >= 1.12.14
      - pip install -I sh=1.12.14
    * - filelock
      - >= 3.0.0
      - pip install -I filelock=3.0.0
    * - docker (for caffe)
      - >= 17.09.0-ce
      - `install doc <https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository>`__

Docker Images
----------------

* Login in `Xiaomi Docker Registry <http://docs.api.xiaomi.net/docker-registry/>`__

.. code:: sh

    docker login cr.d.xiaomi.net

* Build with Dockerfile

.. code:: sh

    docker build -t cr.d.xiaomi.net/mace/mace-dev


* Pull image from docker registry

.. code:: sh

    docker pull cr.d.xiaomi.net/mace/mace-dev

* Create container

.. code:: sh

    # Set 'host' network to use ADB
    docker run -it --rm -v /local/path:/container/path --net=host cr.d.xiaomi.net/mace/mace-dev /bin/bash


使用简介
--------

=======================
1. 获取最新tag的代码
=======================

.. warning::

    建议尽可能使用最新tag下的代码。

.. code:: sh

    git clone git@v9.git.n.xiaomi.com:deep-computing/mace.git

    # update
    git fetch --all --tags --prune

    # get latest tag version
    tag_name=`git describe --abbrev=0 --tags`

    # checkout to latest tag branch
    git checkout -b ${tag_name} tags/${tag_name}

==================
2. 模型优化
==================

-  Tensorflow

TensorFlow训练得到的模型进行一系列的转换，可以提升设备上的运行速度。TensorFlow提供了官方工具
`TensorFlow Graph Transform
Tool <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md>`__
来进行模型优化
(此工具Docker镜像中已经提供，也可以直接点击
`transform_graph <http://cnbj1-inner-fds.api.xiaomi.net/mace/tool/transform_graph>`__
下载这个工具，用户亦可从官方源码编译)。以下分别是GPU模型和DSP模型的优化命令：

.. code:: sh

    # GPU模型:
    ./transform_graph \
        --in_graph=tf_model.pb \
        --out_graph=tf_model_opt.pb \
        --inputs='input' \
        --outputs='output' \
        --transforms='strip_unused_nodes(type=float, shape="1,64,64,3") 
            strip_unused_nodes(type=float, shape="1,64,64,3")
            remove_nodes(op=Identity, op=CheckNumerics)
            fold_constants(ignore_errors=true)
            flatten_atrous_conv
            fold_batch_norms
            fold_old_batch_norms
            strip_unused_nodes
            sort_by_execution_order'

    # DSP模型:
    ./transform_graph \
        --in_graph=tf_model.pb \
        --out_graph=tf_model_opt.pb \
        --inputs='input' \
        --outputs='output' \
        --transforms='strip_unused_nodes(type=float, shape="1,64,64,3") 
            strip_unused_nodes(type=float, shape="1,64,64,3")
            remove_nodes(op=Identity, op=CheckNumerics)
            fold_constants(ignore_errors=true)
            fold_batch_norms
            fold_old_batch_norms
            backport_concatv2
            quantize_weights(minimum_size=2)
            quantize_nodes
            strip_unused_nodes
            sort_by_execution_order'

-  Caffe

Caffe目前只支持最新版本，旧版本请使用Caffe的工具进行升级。

.. code:: bash

    # Upgrade prototxt
    $CAFFE_ROOT/build/tools/upgrade_net_proto_text MODEL.prototxt MODEL.new.prototxt

    # Upgrade caffemodel
    $CAFFE_ROOT/build/tools/upgrade_net_proto_binary MODEL.caffemodel MODEL.new.caffemodel

==================
3. 生成模型静态库
==================

---------------------------------------
3.1 简介
---------------------------------------

Mace目前只提供静态库，有以下两种使用场景。

**特定SOC库**

    该使用场景要求在``yaml``文件中必须制定``target_socs``。主要用于为编译适用于指定手机SOC的静态库。
    如果希望使用GPU，那么编译过程会自动测试选择最佳的GPU相关参数以获得更好的性能。

    .. warning::

         该场景下，你必须插入符合SOC的手机。

**通用库**

    如果在``yaml``文件中没有指定``target_soc``，生成的静态库适用于所有手机。

    .. warning::

         该场景下，GPU性能会略逊于第一种场景。


我们提供了\ ``tools/converter.py``\ 工具，用于编译和运行。

.. warning::

     必须在mace项目的根目录下运行\ ``tools/converter.py``\ 脚本。


---------------------------------------
3.2 \ ``tools/converter.py``\ 脚本
---------------------------------------

**Commands**

    **build**

        .. note::

            build模型静态库以及测试工具。

        * *--config* (type=str,  default="",  required)：模型配置yaml文件路径.
        * *--tuning* (default=false, optional)：是否为特定SOC调制GPU参数.
        * *--enable_openmp* (default=true, optional)：是否启用openmp.

    **run**

        .. note::

            命令行运行模型

        * *--config* (type=str,  default="",  required)：模型配置yaml文件路径.
        * *--round* (type=int, default=1,  optional)：模型运行次数。
        * *--validate* (default=false, optional): 是否需要验证运行结果与框架运行结果是否一致。
        * *--caffe_env* (type=local/docker, default=docker,  optional)：当vaildate时，可以选择指定caffe环境,local表示本地，docker表示使用docker容器.
        * *--restart_round* (type=int, default=1,  optional)：模型重启次数。
        * *--check_gpu_out_of_memory* (default=false, optional): 是否需要检查gpu内存越界。
        * *--vlog_level* (type=int[0-5], default=0,  optional)：详细日志级别.

        .. warning::

            run依赖于build命令.build完成以后才可以执行run命令

    **benchmark**
        * *--config* (type=str,  default="",  required)：模型配置yaml文件路径.

        .. warning::

            benchmark依赖于build命令.

    **通用参数**

    .. list-table::
        :widths: auto
        :header-rows: 1
        :align: left

        * - argument(key)
          - argument(value)
          - default
          - required
          - commands
          - explanation
        * - --omp_num_threads
          - int
          - -1
          - N
          - run/benchmark
          - number of threads
        * - --cpu_affinity_policy
          - int
          - 1
          - N
          - run/benchmark
          - 0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY
        * - --gpu_perf_hint
          - int
          - 3
          - N
          - run/benchmark
          - 0:DEFAULT/1:LOW/2:NORMAL/3:HIGH
        * - --gpu_perf_hint
          - int
          - 3
          - N
          - run/benchmark
          - 0:DEFAULT/1:LOW/2:NORMAL/3:HIGH
        * - --gpu_priority_hint
          - int
          - 3
          - N
          - run/benchmark
          - 0:DEFAULT/1:LOW/2:NORMAL/3:HIGH

---------------------------------------
3.3 \ ``tools/converter.py``\ 使用示例
---------------------------------------
.. code:: sh

    # print help message
    python tools/converter.py -h
    python tools/converter.py build -h
    python tools/converter.py run -h
    python tools/converter.py benchmark -h

    # 仅编译模型和生成静态库
    python tools/converter.py build --config=models/config.yaml

    # 测试模型的运行时间
    python tools/converter.py run --config=models/config.yaml --round=100

    # 对比编译好的模型在mace上与直接使用tensorflow或者caffe运行的结果，相似度使用`余弦距离表示`
    # 其中使用OpenCL设备，默认相似度大于等于`0.995`为通过；DSP设备下，相似度需要达到`0.930`。
    python tools/converter.py run --config=models/config.yaml --validate

    # 模型Benchmark：查看每个Op的运行时间
    python tools/converter.py benchmark --config=models/config.yaml

    # 查看模型运行时占用内存（如果有多个模型，可能需要注释掉一部分配置，只剩一个模型的配置）
    python tools/converter.py run --config=models/config.yaml --round=10000 &
    adb shell dumpsys meminfo | grep mace_run
    sleep 10
    kill %1

==========
4. 发布
==========

``build``命令会生成一个tar包，里面包含了发布所需要的所有文件，其位于``./build/${library_name}/libmace_${library_name}.tar.gz``.
下面解释了该包中包含了哪些文件。

**头文件**
    * ``./build/${library_name}/include/mace/public/*.h``

**静态库**
    * ``./build/${library_name}/library/${target_abi}/*.a``

**动态库**
    * ``./build/${library_name}/library/${target_abi}/libhexagon_controller.so``

    .. note::

        仅编译的模型中包含dsp模式时用到

**模型文件**
    * ``./build/${library_name}/model/${MODEL_TAG}.pb``
    * ``./build/${library_name}/model/${MODEL_TAG}.data``

    .. note::

        pb文件紧当模型build_type设置为proto时才会产生。

**OpenCL预编译文件**
    * ``opencl/${target_abi}/${library_name}_compiled_opencl_kernel.${device_name}.${target_soc}.bin``

    .. note::

        只有指定了``target_soc``并且``runtime==gpu``的情况下才会生成。

    .. warning::

        该文件依赖于手机上opencl驱动，如果OpenCL版本变化，请更新该文件。

**库文件tar包**
    * ``./build/${library_name}/libmace_${library_name}.tar.gz``

    .. note::

        该文件包含了上述所有文件，可以发布使用。

============
5. 使用
============

具体使用流程可参考\ ``mace/examples/mace_run.cc``\ ，下面列出关键步骤。

.. code:: cpp

    // 引入头文件
    #include "mace/public/mace.h"
    #include "mace/public/mace_engine_factory.h"

    // 0. 设置内部存储（设置一次即可）
    const std::string file_path ="/path/to/store/internel/files";
    std::shared_ptr<KVStorageFactory> storage_factory(
        new FileStorageFactory(file_path));
    ConfigKVStorageFactory(storage_factory);

    // 2. 如果你使用特定SOC的GPU，可以设置OpenCL预编译的二进制文件路径。
    //    * 该二进制文件是依赖于手机上OpenCL driver的，如果OpenCL driver改变了，
    //      你需要重新编译并更新该二进制文件。
    if (device_type == DeviceType::GPU) {
      mace::SetOpenCLBinaryPaths(opencl_binary_paths);
    }

    //1. 声明设备类型(必须与build时指定的runtime一致）
    DeviceType device_type = DeviceType::GPU;

    //2. 定义输入输出名称数组
    std::vector<std::string> input_names = {...};
    std::vector<std::string> output_names = {...};

    //3. 创建MaceEngine对象
    std::shared_ptr<mace::MaceEngine> engine;
    MaceStatus create_engine_status;
    // Create Engine from code
    create_engine_status =
        CreateMaceEngineFromCode(model_name.c_str(),
                                 nullptr,
                                 input_names,
                                 output_names,
                                 device_type,
                                 &engine);
    // Create Engine from proto file
    create_engine_status =
        CreateMaceEngineFromProto(model_pb_data,
                                  model_data_file.c_str(),
                                  input_names,
                                  output_names,
                                  device_type,
                                  &engine);
    if (create_engine_status != MaceStatus::MACE_SUCCESS) {
      // do something
    }

    //4. 创建输入输出对象
    std::map<std::string, mace::MaceTensor> inputs;
    std::map<std::string, mace::MaceTensor> outputs;
    for (size_t i = 0; i < input_count; ++i) {
      // Allocate input and output
      int64_t input_size =
          std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 1,
                          std::multiplies<int64_t>());
      auto buffer_in = std::shared_ptr<float>(new float[input_size],
                                              std::default_delete<float[]>());
      // load input
      ...

      inputs[input_names[i]] = mace::MaceTensor(input_shapes[i], buffer_in);
    }

    for (size_t i = 0; i < output_count; ++i) {
      int64_t output_size =
          std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 1,
                          std::multiplies<int64_t>());
      auto buffer_out = std::shared_ptr<float>(new float[output_size],
                                               std::default_delete<float[]>());
      outputs[output_names[i]] = mace::MaceTensor(output_shapes[i], buffer_out);
    }

    //5. 执行模型，得到结果
    engine.Run(inputs, &outputs);

