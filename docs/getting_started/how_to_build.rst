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
    * - Tensorflow
      - >= 1.6.0. (first choice, convenient for Android NN API in the future)
    * - Caffe
      - >= 1.0.

Environment Requirement
-------------------------

``mace``\ supply a docker image which contains all required environment. ``Dockerfile`` under the ``./docker`` directory.
the followings are start commands:

.. code:: sh

    sudo docker pull cr.d.xiaomi.net/mace/mace-dev
    sudo docker run -it --rm --privileged -v /dev/bus/usb:/dev/bus/usb --net=host -v /local/path:/container/path cr.d.xiaomi.net/mace/mace-dev /bin/bash

if you want to run on your local computer, you have to install the following softwares.

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


Usage
--------

============================
1. Pull code with latest tag
============================

.. warning::

    please do not use master branch for deployment.

.. code:: sh

    git clone git@v9.git.n.xiaomi.com:deep-computing/mace.git

    # update
    git fetch --all --tags --prune

    # get latest tag version
    tag_name=`git describe --abbrev=0 --tags`

    # checkout to latest tag branch
    git checkout -b ${tag_name} tags/${tag_name}

============================
2. Model Optimization
============================

-  Tensorflow

Tensorflow supply a
`model optimization tool <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md>`__
for speed up inference. The docker image contain the tool,
by the way you can download from `transform_graph <http://cnbj1-inner-fds.api.xiaomi.net/mace/tool/transform_graph>`__
or compile from tensorflow source code.

The following commands are optimization for CPU, GPU and DSP.

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

Only support versions greater then 1.0, please use the tools caffe supplied to upgrade the models.

.. code:: bash

    # Upgrade prototxt
    $CAFFE_ROOT/build/tools/upgrade_net_proto_text MODEL.prototxt MODEL.new.prototxt

    # Upgrade caffemodel
    $CAFFE_ROOT/build/tools/upgrade_net_proto_binary MODEL.caffemodel MODEL.new.caffemodel

============================
3. Build static library
============================

-----------------
3.1 Overview
-----------------
Mace only build static library. the followings are two use cases.

* **build for specified SOC**

    You must assign ``target_socs`` in yaml configuration file.
    if you want to use gpu for the soc, mace will tuning the parameters for better performance automatically.

    .. warning::

         you should plug in a phone with that soc.

* **build for all SOC**

    When no ``target_soc`` specified, the library is suitable for all soc.

    .. warning::

         The performance will be a little poorer than the first case.

We supply a python script ``tools/converter.py`` to build the library and run the model on the command line.

.. warning::

     must run the script on the root directory of the mace code.


------------------------------------------
3.2 \ ``tools/converter.py``\  explanation
------------------------------------------

**Commands**

    * **build**

        .. note::

            build static library and test tools.

        * *--config* (type=str,  default="",  required)： the path of model yaml configuration file.
        * *--tuning* (default=false, optional)： whether tuning the parameters for the GPU of specified SOC.
        * *--enable_openmp* (default=true, optional)： whether use openmp.

    * **run**

        .. note::

           run the models in command line

        * *--config* (type=str,  default="",  required)： the path of model yaml configuration file.
        * *--round* (type=int, default=1,  optional)： times for run.
        * *--validate* (default=false, optional): whether to verify the results of mace are consistent with the frameworks。
        * *--caffe_env* (type=local/docker, default=docker,  optional)： you can specific caffe environment for validation. local environment or caffe docker image.
        * *--restart_round* (type=int, default=1,  optional)： restart round between run.
        * *--check_gpu_out_of_memory* (default=false, optional): whether check out of memory for gpu.
        * *--vlog_level* (type=int[0-5], default=0,  optional): verbose log level for debug.

        .. warning::

            ``run`` rely on ``build`` command, you should ``run`` after ``build``.

    * **benchmark**
        * *--config* (type=str,  default="",  required)： the path of model yaml configuration file.

        .. warning::

            ``benchmark`` rely on ``build`` command, you should ``benchmark`` after ``build``.

    **common arguments**

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
          - ``run``/``benchmark``
          - number of threads
        * - --cpu_affinity_policy
          - int
          - 1
          - N
          - ``run``/``benchmark``
          - 0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY
        * - --gpu_perf_hint
          - int
          - 3
          - N
          - ``run``/``benchmark``
          - 0:DEFAULT/1:LOW/2:NORMAL/3:HIGH
        * - --gpu_perf_hint
          - int
          - 3
          - N
          - ``run``/``benchmark``
          - 0:DEFAULT/1:LOW/2:NORMAL/3:HIGH
        * - --gpu_priority_hint
          - int
          - 3
          - N
          - ``run``/``benchmark``
          - 0:DEFAULT/1:LOW/2:NORMAL/3:HIGH

---------------------------------------------
3.3 \ ``tools/converter.py``\ usage examples
---------------------------------------------

.. code:: sh

    # print help message
    python tools/converter.py -h
    python tools/converter.py build -h
    python tools/converter.py run -h
    python tools/converter.py benchmark -h

    # Build the static library
    python tools/converter.py build --config=models/config.yaml

    # Test model run time
    python tools/converter.py run --config=models/config.yaml --round=100

    # Compare the results of mace and platform. use the **cosine distance** to represent similarity.
    python tools/converter.py run --config=models/config.yaml --validate

    # Benchmark Model: check the execution time of each Op.
    python tools/converter.py benchmark --config=models/config.yaml

    # Check the memory usage of the model(**Just keep only one model in configuration file**)
    python tools/converter.py run --config=models/config.yaml --round=10000 &
    adb shell dumpsys meminfo | grep mace_run
    sleep 10
    kill %1

=============
4. Deployment
=============

``build`` command will generate a package which contains the static library, model files and header files.
the package is at ``./build/${library_name}/libmace_${library_name}.tar.gz``.
The followings list the details.

**header files**
    * ``include/mace/public/*.h``

**static libraries**
    * ``library/${target_abi}/*.a``

**dynamic libraries**
    * ``library/libhexagon_controller.so``

    .. note::

        only use for DSP

**model files**
    * ``model/${MODEL_TAG}.pb``
    * ``model/${MODEL_TAG}.data``

    .. note::

        ``.pb`` file will be generated only when build_type is ``proto``.

**OpenCL compiled kernel binary file**
    * ``opencl/${target_abi}/${library_name}_compiled_opencl_kernel.${device_name}.${target_soc}.bin``

    .. note::

        This file will be generated only when specify ``target_soc`` and runtime is ``gpu``.

    .. warning::

        This file rely on the OpenCL driver on the phone, you should update the file when OpenCL driver changed.

**tar package**
    * ``./build/${library_name}/libmace_${library_name}.tar.gz``

    .. note::

        This file package all the above files which used for deployment.

=============
5. how to use
=============

Please refer to \ ``mace/examples/example.cc``\ for full usage. the following list the key steps.

.. code:: cpp

    // include the header files
    #include "mace/public/mace.h"
    #include "mace/public/mace_runtime.h"
    #include "mace/public/mace_engine_factory.h"

    // 0. set internal storage factory（**Call once**）
    const std::string file_path ="/path/to/store/internel/files";
    std::shared_ptr<KVStorageFactory> storage_factory(
        new FileStorageFactory(file_path));
    ConfigKVStorageFactory(storage_factory);

    // 1. set precompiled OpenCL binary file paths if you use gpu of specified SOC,
    //    Besides the binary rely on the OpenCL driver of the SOC,
    //    if OpenCL driver changed, you should recompiled the binary file.
    if (device_type == DeviceType::GPU) {
      mace::SetOpenCLBinaryPaths(opencl_binary_paths);
    }

    // 2. Declare the device type(must be same with ``runtime`` in configuration file)
    DeviceType device_type = DeviceType::GPU;

    // 3. Define the input and output tensor names.
    std::vector<std::string> input_names = {...};
    std::vector<std::string> output_names = {...};

    // 4. Create MaceEngine object
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

    // 5. Create Input and Output objects
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

    // 6. Run the model
    MaceStatus status = engine.Run(inputs, &outputs);

