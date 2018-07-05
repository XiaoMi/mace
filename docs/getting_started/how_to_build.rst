How to build
============

Supported Platforms
-------------------

.. list-table::
    :widths: auto
    :header-rows: 1
    :align: left

    * - Platform
      - Explanation
    * - TensorFlow
      - >= 1.6.0.
    * - Caffe
      - >= 1.0.

Environment Requirement
-------------------------

MACE requires the following dependencies:

.. list-table::
    :widths: auto
    :header-rows: 1
    :align: left

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
    * - tensorflow
      - >= 1.6.0
      - pip install -I tensorflow==1.6.0 (if you use tensorflow model)
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

MACE provides a Dockerfile with these dependencies installed,
you can build the image from it,

.. code:: sh

    docker build -t registry.cn-hangzhou.aliyuncs.com/xiaomimace/mace-dev-lite ./docker/mace-dev-lite

or pull the pre-built image from Docker Hub,

.. code:: sh

    docker pull registry.cn-hangzhou.aliyuncs.com/xiaomimace/mace-dev-lite

and then run the container with the following command.

.. code:: sh

    # Create container
    # Set 'host' network to use ADB
    docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb --net=host \
               -v /local/path:/container/path \
               registry.cn-hangzhou.aliyuncs.com/xiaomimace/mace-dev-lite \
               /bin/bash


Usage
--------

=======================================
1. Pull MACE source code
=======================================

.. code:: sh

    git clone https://github.com/XiaoMi/mace.git
    git fetch --all --tags --prune

    # Checkout the latest tag (i.e. release version)
    tag_name=`git describe --abbrev=0 --tags`
    git checkout tags/${tag_name}

.. note::

    It's highly recommanded to use a release version instead of master branch.

============================
2. Model Preprocessing
============================

-  TensorFlow

TensorFlow provides 
`Graph Transform Tool <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md>`__
to improve inference efficiency by making various optimizations like Ops
folding, redundant node removal etc. It's strongly recommended to make these
optimizations before graph conversion step.

The following commands show the suggested graph transformations and
optimizations for different runtimes,

.. code:: sh

    # CPU/GPU:
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

.. code:: sh

    # DSP:
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

MACE converter only supports Caffe 1.0+, you need to upgrade
your models with Caffe built-in tool when necessary,

.. code:: bash

    # Upgrade prototxt
    $CAFFE_ROOT/build/tools/upgrade_net_proto_text MODEL.prototxt MODEL.new.prototxt

    # Upgrade caffemodel
    $CAFFE_ROOT/build/tools/upgrade_net_proto_binary MODEL.caffemodel MODEL.new.caffemodel

==============================
3. Build static/shared library
==============================

-----------------
3.1 Overview
-----------------
MACE can build either static or shared library (which is
specified by ``linkshared`` in YAML model deployment file).
The followings are two use cases.

* **Build well tuned library for specific SoCs**

    When ``target_socs`` is specified in YAML model deployment file, the build
    tool will enable automatic tuning for GPU kernels. This usually takes some
    time to finish depending on the complexity of your model.

    .. note::

         You should plug in device(s) with the correspoding SoC(s).

* **Build generic library for all SoCs**

    When ``target_socs`` is not specified, the generated library is compatible
    with general devices.

    .. note::

         There will be around of 1 ~ 10% performance drop for GPU
         runtime compared to the well tuned library.

MACE provide command line tool (``tools/converter.py``) for
model conversion, compiling, test run, benchmark and correctness validation.

.. note::

     1. ``tools/converter.py`` should be run at the root directory of this project.
     2. When ``linkshared`` is set to ``1``, ``build_type`` should be ``proto``.
        And currently only android devices supported.


------------------------------------------
3.2 \ ``tools/converter.py``\  usage
------------------------------------------

**Commands**

    * **build**

        build library and test tools.

    .. code:: sh

        # Build library 
        python tools/converter.py build --config=models/config.yaml



    * **run**

        run the model(s).

    .. code:: sh

    	# Test model run time
        python tools/converter.py run --config=models/config.yaml --round=100

    	# Validate the correctness by comparing the results against the
    	# original model and framework, measured with cosine distance for similarity.
    	python tools/converter.py run --config=models/config.yaml --validate

    	# Check the memory usage of the model(**Just keep only one model in configuration file**)
    	python tools/converter.py run --config=models/config.yaml --round=10000 &
    	sleep 5
    	adb shell dumpsys meminfo | grep mace_run
    	kill %1


    .. warning::

        ``run`` rely on ``build`` command, you should ``run`` after ``build``.

    * **benchmark**

        benchmark and profiling model.

    .. code:: sh

        # Benchmark model, get detailed statistics of each Op.
        python tools/converter.py benchmark --config=models/config.yaml


    .. warning::

        ``benchmark`` rely on ``build`` command, you should ``benchmark`` after ``build``.

**Common arguments**

    .. list-table::
        :widths: auto
        :header-rows: 1
        :align: left

        * - option
          - type
          - default
          - commands
          - explanation
        * - --omp_num_threads
          - int
          - -1
          - ``run``/``benchmark``
          - number of threads
        * - --cpu_affinity_policy
          - int
          - 1
          - ``run``/``benchmark``
          - 0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY
        * - --gpu_perf_hint
          - int
          - 3
          - ``run``/``benchmark``
          - 0:DEFAULT/1:LOW/2:NORMAL/3:HIGH
        * - --gpu_perf_hint
          - int
          - 3
          - ``run``/``benchmark``
          - 0:DEFAULT/1:LOW/2:NORMAL/3:HIGH
        * - --gpu_priority_hint
          - int
          - 3
          - ``run``/``benchmark``
          - 0:DEFAULT/1:LOW/2:NORMAL/3:HIGH

Using ``-h`` to get detailed help.

.. code:: sh

    python tools/converter.py -h
    python tools/converter.py build -h
    python tools/converter.py run -h
    python tools/converter.py benchmark -h


=============
4. Deployment
=============

``build`` command will generate the static/shared library, model files and
header files and package them as
``build/${library_name}/libmace_${library_name}.tar.gz``.

-  The generated ``static`` libraries are organized as follows,

.. code::

      build/
      └── mobilenet-v2-gpu
          ├── include
          │   └── mace
          │       └── public
          │           ├── mace.h
          │           └── mace_runtime.h
          ├── libmace_mobilenet-v2-gpu.tar.gz
          ├── lib
          │   ├── arm64-v8a
          │   │   └── libmace_mobilenet-v2-gpu.MI6.msm8998.a
          │   └── armeabi-v7a
          │       └── libmace_mobilenet-v2-gpu.MI6.msm8998.a
          ├── model
          │   ├── mobilenet_v2.data
          │   └── mobilenet_v2.pb
          └── opencl
              ├── arm64-v8a
              │   └── mobilenet-v2-gpu_compiled_opencl_kernel.MI6.msm8998.bin
              └── armeabi-v7a
                  └── mobilenet-v2-gpu_compiled_opencl_kernel.MI6.msm8998.bin

-  The generated ``shared`` libraries are organized as follows,

.. code::

      build
      └── mobilenet-v2-gpu
          ├── include
          │   └── mace
          │       └── public
          │           ├── mace.h
          │           └── mace_runtime.h
          ├── lib
          │   ├── arm64-v8a
          │   │   ├── libgnustl_shared.so
          │   │   └── libmace.so
          │   └── armeabi-v7a
          │       ├── libgnustl_shared.so
          │       └── libmace.so
          ├── model
          │   ├── mobilenet_v2.data
          │   └── mobilenet_v2.pb
          └── opencl
              ├── arm64-v8a
              │   └── mobilenet-v2-gpu_compiled_opencl_kernel.MI6.msm8998.bin
              └── armeabi-v7a
                  └── mobilenet-v2-gpu_compiled_opencl_kernel.MI6.msm8998.bin

.. note::

    1. DSP runtime depends on ``libhexagon_controller.so``.
    2. ``${MODEL_TAG}.pb`` file will be generated only when ``build_type`` is ``proto``.
    3. ``${library_name}_compiled_opencl_kernel.${device_name}.${soc}.bin`` will
       be generated only when ``target_socs`` and ``gpu`` runtime are specified.
    4. Generated shared library depends on ``libgnustl_shared.so``.

.. warning::

    ``${library_name}_compiled_opencl_kernel.${device_name}.${soc}.bin`` depends
    on the OpenCL version of the device, you should maintan the compatibility or
    configure compiling cache store with ``ConfigKVStorageFactory``.

=========================================
5. How to use the library in your project
=========================================

Please refer to \ ``mace/examples/example.cc``\ for full usage. The following list the key steps.

.. code:: cpp

    // Include the headers
    #include "mace/public/mace.h"
    #include "mace/public/mace_runtime.h"
    // If the build_type is code
    #include "mace/public/mace_engine_factory.h"

    // 0. Set pre-compiled OpenCL binary program file paths when available
    if (device_type == DeviceType::GPU) {
      mace::SetOpenCLBinaryPaths(opencl_binary_paths);
    }

    // 1. Set compiled OpenCL kernel cache, this is used to reduce the
    // initialization time since the compiling is too slow. It's suggested
    // to set this even when pre-compiled OpenCL program file is provided
    // because the OpenCL version upgrade may also leads to kernel
    // recompilations.
    const std::string file_path ="path/to/opencl_cache_file";
    std::shared_ptr<KVStorageFactory> storage_factory(
        new FileStorageFactory(file_path));
    ConfigKVStorageFactory(storage_factory);

    // 2. Declare the device type (must be same with ``runtime`` in configuration file)
    DeviceType device_type = DeviceType::GPU;

    // 3. Define the input and output tensor names.
    std::vector<std::string> input_names = {...};
    std::vector<std::string> output_names = {...};

    // 4. Create MaceEngine instance 
    std::shared_ptr<mace::MaceEngine> engine;
    MaceStatus create_engine_status;
    // Create Engine from compiled code
    create_engine_status =
        CreateMaceEngineFromCode(model_name.c_str(),
                                 nullptr,
                                 input_names,
                                 output_names,
                                 device_type,
                                 &engine);
    // Create Engine from model file
    create_engine_status =
        CreateMaceEngineFromProto(model_pb_data,
                                  model_data_file.c_str(),
                                  input_names,
                                  output_names,
                                  device_type,
                                  &engine);
    if (create_engine_status != MaceStatus::MACE_SUCCESS) {
      // Report error
    }

    // 5. Create Input and Output tensor buffers
    std::map<std::string, mace::MaceTensor> inputs;
    std::map<std::string, mace::MaceTensor> outputs;
    for (size_t i = 0; i < input_count; ++i) {
      // Allocate input and output
      int64_t input_size =
          std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 1,
                          std::multiplies<int64_t>());
      auto buffer_in = std::shared_ptr<float>(new float[input_size],
                                              std::default_delete<float[]>());
      // Load input here
      // ...

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

    // 6. Run the model
    MaceStatus status = engine.Run(inputs, &outputs);

