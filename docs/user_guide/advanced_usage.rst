Advanced usage
===============

This part contains the full usage of MACE.


How to build
-------------


=========
Overview
=========

As mentioned in the previous part, a model deployment file defines a case of model deployment.
The whole building process is loading a deployment file, converting models, building MACE and packing generated files.

================
Deployment file
================


One deployment file will generate one library normally, but if more than one ABIs are specified,
one library will be generated for each ABI.
A deployment file can also contain multiple models. For example, an AI camera application may
contain face recognition, object recognition, and voice recognition models, all of which can be defined
in one deployment file.

* **Example**

    Here is an example deployment file used by an Android demo application.

    .. literalinclude:: models/demo_app_models.yml
        :language: yaml


* **Configurations**


.. list-table::
    :header-rows: 1

    * - Options
      - Usage
    * - library_name
      - Library name.
    * - target_abis
      - The target ABI(s) to build, could be 'host', 'armeabi-v7a' or 'arm64-v8a'.
        If more than one ABIs will be used, seperate them by comas.
    * - target_socs
      - [optional] Build for specific SoCs.
    * - embed_model_data
      - Whether embedding model weights into the code, default is 0.
    * - build_type
      - model build type, can be 'proto' or 'code'. 'proto' for converting model to ProtoBuf file and 'code' for converting model to c++ code.
    * - linkshared
      - [optional] 1 for building shared library, and 0 for static library, default to 0.
    * - model_name
      - model name, should be unique if there are more than one models.
        **LIMIT: if build_type is code, model_name will be used in c++ code so that model_name must comply with c++ name specification.**
    * - platform
      - The source framework, tensorflow or caffe.
    * - model_file_path
      - The path of your model file, can be local path or remote url.
    * - model_sha256_checksum
      - The SHA256 checksum of the model file.
    * - weight_file_path
      - [optional] The path of Caffe model weights file.
    * - weight_sha256_checksum
      - [optional] The SHA256 checksum of Caffe model weights file.
    * - subgraphs
      - subgraphs key. **DO NOT EDIT**
    * - input_tensors
      - The input tensor name(s) (tensorflow) or top name(s) of inputs' layer (caffe).
        If there are more than one tensors, use one line for a tensor.
    * - output_tensors
      - The output tensor name(s) (tensorflow) or top name(s) of outputs' layer (caffe).
        If there are more than one tensors, use one line for a tensor.
    * - input_shapes
      - The shapes of the input tensors, in NHWC order.
    * - output_shapes
      - The shapes of the output tensors, in NHWC order.
    * - input_ranges
      - The numerical range of the input tensors' data, default [-1, 1]. It is only for test.
    * - validation_inputs_data
      - [optional] Specify Numpy validation inputs. When not provided, [-1, 1] random values will be used.
    * - runtime
      - The running device, one of [cpu, gpu, dsp, cpu_gpu]. cpu_gpu contains CPU and GPU model definition so you can run the model on both CPU and GPU.
    * - data_type
      - [optional] The data type used for specified runtime. [fp16_fp32, fp32_fp32] for GPU, default is fp16_fp32, [fp32] for CPU and [uint8] for DSP.
    * - limit_opencl_kernel_time
      - [optional] Whether splitting the OpenCL kernel within 1 ms to keep UI responsiveness, default is 0.
    * - nnlib_graph_mode
      - [optional] Control the DSP precision and performance, default to 0 usually works for most cases.
    * - obfuscate
      - [optional] Whether to obfuscate the model operator name, default to 0.
    * - winograd
      - [optional] Whether to enable Winograd convolution, **will increase memory consumption**.


.. note::

    Some command tools:

    .. code:: bash

        # command for fetching android device's soc info.
        adb shell getprop | grep "model\|version.sdk\|manufacturer\|hardware\|platform\|brand"

        # command for generating sha256_sum
        sha256sum /path/to/your/file


=========
Building
=========

* **Build static or shared library**

    MACE can build either static or shared library (which is
    specified by ``linkshared`` in YAML model deployment file).
    The followings are two using cases.

* **Build well tuned library for specific SoCs**

    When ``target_socs`` is specified in YAML model deployment file, the build
    tool will enable automatic tuning for GPU kernels. This usually takes some
    time to finish depending on the complexity of your model.

    .. note::

         1. You should plug in device(s) with the specific SoC(s).

* **Build generic library for all SoCs**

    When ``target_socs`` is not specified, the generated library is compatible
    with general devices.

    .. note::

         1. There will be around of 1 ~ 10% performance drop for GPU
            runtime compared to the well tuned library.

* **Build models into file or code**

    When ``build_type`` is set to ``code``, model's graph and weights data will be embedded into codes.
    This is used for model protection.

    .. note::

         1. When ``linkshared`` is set to ``1``, ``build_type`` should be ``proto``.
            And currently only android devices supported.
         2. Another model protection method is using ``obfuscate`` to obfuscate the model operator name.


**Commands**

    * **build library and test tools**

    .. code:: sh

        # Build library
        python tools/converter.py build --config=/path/to/model_deployment_file.yml



    * **run the model**

    .. code:: sh

    	# Test model run time
        python tools/converter.py run --config=/path/to/model_deployment_file.yml --round=100

    	# Validate the correctness by comparing the results against the
    	# original model and framework, measured with cosine distance for similarity.
    	python tools/converter.py run --config=/path/to/model_deployment_file.yml --validate

    	# Check the memory usage of the model(**Just keep only one model in configuration file**)
    	python tools/converter.py run --config=/path/to/model_deployment_file.yml --round=10000 &
    	sleep 5
    	adb shell dumpsys meminfo | grep mace_run
    	kill %1


    .. warning::

        ``run`` rely on ``build`` command, you should ``run`` after ``build``.

    * **benchmark and profiling model**

    .. code:: sh

        # Benchmark model, get detailed statistics of each Op.
        python tools/converter.py benchmark --config=/path/to/model_deployment_file.yml


    .. warning::

        ``benchmark`` rely on ``build`` command, you should ``benchmark`` after ``build``.

**Common arguments**

    .. list-table::
        :header-rows: 1

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

Use ``-h`` to get detailed help.

.. code:: sh

    python tools/converter.py -h
    python tools/converter.py build -h
    python tools/converter.py run -h
    python tools/converter.py benchmark -h



How to deploy
--------------


=========
Overview
=========

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
          |           └── mace_engine_factory.h (Only exists if ``build_type`` set to ``code``))
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
          |           └── mace_engine_factory.h (Only exists if ``build_type`` set to ``code``)
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
    5. Files in opencl folder will be generated only if
       ``target_soc`` was set and ``runtime`` contains ``gpu`` in the deployment file.
    6. When ``build_type`` has been set to ``code``, ${library_name}.h and mace_engine_factory.h
       will be generated in ``include`` folder. This header file will be used to create mace_engine of your model.


.. warning::

    ``${library_name}_compiled_opencl_kernel.${device_name}.${soc}.bin`` depends
    on the OpenCL version of the device, you should maintan the compatibility or
    configure compiling cache store with ``ConfigKVStorageFactory``.


===========
Deployment
===========

Unpack the generated libmace_${library_name}.tar.gz file and copy all of the uncompressed files into your project.

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

