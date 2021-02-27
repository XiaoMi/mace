Advanced usage for Bazel users
================================

This part contains the full usage of MACE.

Overview
---------

As mentioned in the previous part, a model deployment file defines a case of model deployment.
The building process includes parsing model deployment file, converting models,
building MACE core library and packing generated model libraries.

Deployment file
---------------


One deployment file will generate one library normally, but if more than one ABIs are specified,
one library will be generated for each ABI.
A deployment file can also contain multiple models. For example, an AI camera application may
contain face recognition, object recognition, and voice recognition models, all of which can be defined
in one deployment file.

* **Example**

    Here is an example deployment file with two models.

    .. literalinclude:: models/demo_models.yml
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
        If more than one ABIs will be used, separate them by commas.
    * - target_socs
      - [optional] Build for specific SoCs.
    * - model_graph_format
      - model graph format, could be 'file' or 'code'. 'file' for converting model graph to ProtoBuf file(.pb) and 'code' for converting model graph to c++ code.
    * - model_data_format
      - model data format, could be 'file' or 'code'. 'file' for converting model weight to data file(.data) and 'code' for converting model weight to c++ code.
    * - model_name
      - model name should be unique if there are more than one models.
        **LIMIT: if build_type is code, model_name will be used in c++ code so that model_name must comply with c++ name specification.**
    * - platform
      - The source framework, tensorflow or caffe.
    * - model_file_path
      - The path of your model file which can be local path or remote URL.
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
      - The shapes of the input tensors, default is NHWC order.
    * - output_shapes
      - The shapes of the output tensors, default is NHWC order.
    * - input_ranges
      - The numerical range of the input tensors' data, default [-1, 1]. It is only for test.
    * - validation_inputs_data
      - [optional] Specify Numpy validation inputs. When not provided, [-1, 1] random values will be used.
    * - accuracy_validation_script
      - [optional] Specify the accuracy validation script as a plugin to test accuracy, see `doc <#validate-accuracy-of-mace-model>`__.
    * - validation_threshold
      - [optional] Specify the similarity threshold for validation. A dict with key in 'CPU', 'GPU' and/or 'HEXAGON' and value <= 1.0.
    * - backend
      - The onnx backend framework for validation, could be [tensorflow, caffe2, pytorch], default is tensorflow.
    * - runtime
      - The running device, one of [cpu, gpu, dsp, cpu+gpu]. cpu+gpu contains CPU and GPU model definition so you can run the model on both CPU and GPU.
    * - data_type
      - [optional] The data type used for specified runtime. [fp16_fp32, fp32_fp32] for GPU and APU; [fp16_fp32, bf16_fp32, fp32_fp32, fp16_fp16] for CPU, default is fp16_fp32.
    * - input_data_types
      - [optional] The input data type for specific op(eg. gather), which can be [int32, float32], default to float32.
    * - input_data_formats
      - [optional] The format of the input tensors, one of [NONE, NHWC, NCHW]. If there is no format of the input, please use NONE. If only one single format is specified, all inputs will use that format, default is NHWC order.
    * - output_data_formats
      - [optional] The format of the output tensors, one of [NONE, NHWC, NCHW]. If there is no format of the output, please use NONE. If only one single format is specified, all inputs will use that format, default is NHWC order.
    * - limit_opencl_kernel_time
      - [optional] Whether splitting the OpenCL kernel within 1 ms to keep UI responsiveness, default is 0.
    * - opencl_queue_window_size
      - [optional] Limit the max commands in OpenCL command queue to keep UI responsiveness, default is 0.
    * - obfuscate
      - [optional] Whether to obfuscate the model operator name, default to 0.
    * - winograd
      - [optional] Which type winograd to use, could be [0, 2, 4]. 0 for disable winograd, 2 and 4 for enable winograd, 4 may be faster than 2 but may take more memory.


.. note::

    Some command tools:

    .. code-block:: bash

        # Get device's soc info.
        adb shell getprop | grep platform

        # command for generating sha256_sum
        sha256sum /path/to/your/file



Advanced usage
--------------

There are three common advanced use cases:
  - run your model on the embedded device(ARM LINUX)
  - converting model to C++ code.
  - tuning GPU kernels for a specific SoC.

Run you model on the embedded device(ARM Linux)
-----------------------------------------------

The way to run your model on the ARM Linux is nearly same as with android, except you need specify a device config file.

.. code-block:: bash

    python tools/converter.py run --config=/path/to/your/model_deployment_file.yml --device_yml=/path/to/devices.yml

There are two steps to do before run:

1. configure login without password

    MACE use ssh to connect embedded device, you should copy your public key to embedded device with the blow command.

    .. code-block:: bash

      cat ~/.ssh/id_rsa.pub | ssh -q {user}@{ip} "cat >> ~/.ssh/authorized_keys"

2. write your own device yaml configuration file.

    * **Example**

        Here is an device yaml config demo.

        .. literalinclude:: devices/demo_device_nanopi.yml
            :language: yaml

    * **Configuration**
        The detailed explanation is listed in the blow table.

        .. list-table::
            :header-rows: 1

            * - Options
              - Usage
            * - target_abis
              - Device supported abis, you can get it via ``dpkg --print-architecture`` and
                ``dpkg --print-foreign-architectures`` command, if more than one abi is supported,
                separate them by commas.
            * - target_socs
              - device soc, you can get it from device manual, we haven't found a way to get it in shell.
            * - models
              - device models full name, you can get via get ``lshw`` command (third party package, install it via your package manager).
                see it's product value.
            * - address
              - Since we use ssh to connect device, ip address is required.
            * - username
              - login username, required.


Convert model(s) to C++ code
----------------------------

    * **1. Change the model deployment file(.yml)**

        If you want to protect your model, you can convert model to C++ code. there are also two cases:

        * convert model graph to code and model weight to file with below model configuration.

        .. code-block:: sh

            model_graph_format: code
            model_data_format: file

        * convert both model graph and model weight to code with below model configuration.

        .. code-block:: sh

            model_graph_format: code
            model_data_format: code

        .. note::

             Another model protection method is using ``obfuscate`` to obfuscate names of model's operators.

    * **2. Convert model(s) to code**

        .. code-block:: sh

            python tools/converter.py convert --config=/path/to/model_deployment_file.yml

        The command will generate **${library_name}.a** in **build/${library_name}/model** directory and
        ** *.h ** in **build/${library_name}/include** like the following dir-tree.

        .. code-block:: none

             # model_graph_format: code
             # model_data_format: file

             build
               ├── include
               │   └── mace
               │       └── public
               │           ├── mace_engine_factory.h
               │           └── mobilenet_v1.h
               └── model
                   ├── mobilenet-v1.a
                   └── mobilenet_v1.data

             # model_graph_format: code
             # model_data_format: code

             build
               ├── include
               │   └── mace
               │       └── public
               │           ├── mace_engine_factory.h
               │           └── mobilenet_v1.h
               └── model
                   └── mobilenet-v1.a

    * **3. Deployment**
        * Link `libmace.a` and `${library_name}.a` to your target.
        * Refer to \ ``mace/tools/mace_run.cc``\ for full usage. The following list the key steps.

        .. code-block:: cpp

            // Include the headers
            #include "mace/public/mace.h"
            // If the model_graph_format is code
            #include "mace/public/${model_name}.h"
            #include "mace/public/mace_engine_factory.h"

            // ... Same with the code in basic usage

            // 4. Create MaceEngine instance
            std::shared_ptr<mace::MaceEngine> engine;
            MaceStatus create_engine_status;
            // Create Engine from compiled code
            create_engine_status =
                CreateMaceEngineFromCode(model_name.c_str(),
                                         model_data_ptr, // nullptr if model_data_format is code
                                         model_data_size, // 0 if model_data_format is code
                                         input_names,
                                         output_names,
                                         device_type,
                                         &engine);
            if (create_engine_status != MaceStatus::MACE_SUCCESS) {
              // Report error or fallback
            }

            // ... Same with the code in basic usage


Transform models after conversion
---------------------------------

    If ``model_graph_format`` or ``model_data_format`` is specified as `file`, the model or weight file will
    be generated as a `.pb` or `.data` file after model conversion. After that, more transformations can be
    applied to the generated files, such as compression or encryption. To achieve that, the model loading is
    split to two stages: 1) load the file from file system to memory buffer; 2) create the MACE engine from the
    model buffer. So between the two stages, transformations can be inserted to decompress or decrypt the model
    buffer. The transformations are user defined. The following lists the key steps when both ``model_graph_format``
    and ``model_data_format`` are set as `file`.

    .. code-block:: cpp

        // Load model graph from file system
        std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_graph_data =
            make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
        if (FLAGS_model_file != "") {
          auto fs = GetFileSystem();
          status = fs->NewReadOnlyMemoryRegionFromFile(FLAGS_model_file.c_str(),
              &model_graph_data);
          if (status != MaceStatus::MACE_SUCCESS) {
            // Report error or fallback
          }
        }
        // Load model data from file system
        std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_weights_data =
            make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
        if (FLAGS_model_data_file != "") {
          auto fs = GetFileSystem();
          status = fs->NewReadOnlyMemoryRegionFromFile(FLAGS_model_data_file.c_str(),
              &model_weights_data);
          if (status != MaceStatus::MACE_SUCCESS) {
            // Report error or fallback
          }
        }
        if (model_graph_data == nullptr || model_weights_data == nullptr) {
          // Report error or fallback
        }

        std::vector<unsigned char> transformed_model_graph_data;
        std::vector<unsigned char> transformed_model_weights_data;
        // Add transformations here.
        ...
        // Release original model data after transformations
        model_graph_data.reset();
        model_weights_data.reset();

        // Create the MACE engine from the model buffer
        std::shared_ptr<mace::MaceEngine> engine;
        MaceStatus create_engine_status;
        create_engine_status =
            CreateMaceEngineFromProto(transformed_model_graph_data.data(),
                                      transformed_model_graph_data.size(),
                                      transformed_model_weights_data.data(),
                                      transformed_model_weights_data.size(),
                                      input_names,
                                      output_names,
                                      config,
                                      &engine);
        if (create_engine_status != MaceStatus::MACE_SUCCESS) {
          // Report error or fallback
        }


Tuning for specific SoC's GPU
-----------------------------

    If you want to use the GPU of a specific device, you can just specify the ``target_socs`` in your YAML file and
    then tune the MACE lib for it (OpenCL kernels), which may get 1~10% performance improvement.

    * **1. Change the model deployment file(.yml)**

        Specify ``target_socs`` in your model deployment file(.yml):

        .. code-block:: sh

            target_socs: [sdm845]

        .. note::

            Get device's soc info: `adb shell getprop | grep platform`

    * **2. Convert model(s)**

        .. code-block:: sh

            python tools/converter.py convert --config=/path/to/model_deployment_file.yml

    * **3. Tuning**

        The tools/converter.py will enable automatic tuning for GPU kernels. This usually takes some
        time to finish depending on the complexity of your model.

        .. note::

             You must specify the ``target_socs`` in your YAML file and plug in device(s) with the specific SoC(s).


        .. code-block:: sh

            python tools/converter.py run --config=/path/to/model_deployment_file.yml

        The command will generate two files in `build/${library_name}/opencl`, like the following dir-tree.

        .. code-block:: none

              build
              └── mobilenet-v2
                  ├── model
                  │   ├── mobilenet_v2.data
                  │   └── mobilenet_v2.pb
                  └── opencl
                      └── arm64-v8a
                         ├── moblinet-v2_compiled_opencl_kernel.MiNote3.sdm660.bin
                         ├── moblinet-v2_compiled_opencl_kernel.MiNote3.sdm660.bin.cc
                         ├── moblinet-v2_tuned_opencl_parameter.MiNote3.sdm660.bin
                         └── moblinet-v2_tuned_opencl_parameter.MiNote3.sdm660.bin.cc


        * **mobilenet-v2-gpu_compiled_opencl_kernel.MI6.msm8998.bin** stands for the OpenCL binaries
          used for your models, which could accelerate the initialization stage.
          Details please refer to `OpenCL Specification <https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clCreateProgramWithBinary.html>`__.
        * **mobilenet-v2-gpu_compiled_opencl_kernel.MI6.msm8998.bin.cc** contains C++ source code which defines OpenCL binary data as const array.
        * **mobilenet-v2-tuned_opencl_parameter.MI6.msm8998.bin** stands for the tuned OpenCL parameters
          for the SoC.
        * **mobilenet-v2-tuned_opencl_parameter.MI6.msm8998.bin.cc** contains C++ source code which defines OpenCL binary data as const array.

    * **4. Deployment**
        * Change the names of files generated above for not collision and push them to **your own device's directory**.
        * Use like the previous procedure, below lists the key steps differently.

        .. code-block:: cpp

            // Include the headers
            #include "mace/public/mace.h"
            // 0. Declare the device type (must be same with ``runtime`` in configuration file)
            DeviceType device_type = DeviceType::GPU;

            // 1. configuration
            MaceStatus status;
            MaceEngineConfig config(device_type);
            std::shared_ptr<GPUContext> gpu_context;

            const std::string storage_path ="path/to/storage";
            gpu_context = GPUContextBuilder()
                .SetStoragePath(storage_path)
                .SetOpenCLBinaryPaths(path/to/opencl_binary_paths)
                .SetOpenCLParameterPath(path/to/opencl_parameter_file)
                .Finalize();
            config.SetGPUContext(gpu_context);
            config.SetGPUHints(
                static_cast<GPUPerfHint>(GPUPerfHint::PERF_NORMAL),
                static_cast<GPUPriorityHint>(GPUPriorityHint::PRIORITY_LOW));

            // ... Same with the code in basic usage.


Validate accuracy of MACE model
-------------------------------

MACE supports **python validation script** as a plugin to test the accuracy, the plugin script could be used for below two purpose.

1. Test the **accuracy(like Top-1)** of MACE model(specifically quantization model) converted from other framework(like tensorflow)
2. Show some real output if you want to see it.

The script define some interfaces like `preprocess` and `postprocess` to deal with input/outut and calculate the accuracy,
you could refer to the `sample code <https://github.com/XiaoMi/mace/tree/master/tools/accuracy_validator.py>`__ for detail.
the sample code show how to calculate the Top-1 accuracy with imagenet validation dataset.


Useful Commands
---------------
* **run the model**

.. code-block:: sh

    # Test model run time
    python tools/converter.py run --config=/path/to/model_deployment_file.yml --round=100

    # Validate the correctness by comparing the results against the
    # original model and framework, measured with cosine distance for similarity.
    python tools/converter.py run --config=/path/to/model_deployment_file.yml --validate

    # Check the memory usage of the model(**Just keep only one model in deployment file**)
    python tools/converter.py run --config=/path/to/model_deployment_file.yml --round=10000 &
    sleep 5
    adb shell dumpsys meminfo | grep mace_run
    kill %1


.. warning::

    ``run`` rely on ``convert`` command, you should ``convert`` before ``run``.

* **benchmark and profile model**

the detailed information is in :doc:`benchmark`.

.. code-block:: sh

    # Benchmark model, get detailed statistics of each Op.
    python tools/converter.py run --config=/path/to/model_deployment_file.yml --benchmark


.. warning::

    ``benchmark`` rely on ``convert`` command, you should ``benchmark`` after ``convert``.

**Common arguments**

    .. list-table::
        :header-rows: 1

        * - option
          - type
          - default
          - commands
          - explanation
        * - --num_threads
          - int
          - -1
          - ``run``
          - number of threads
        * - --cpu_affinity_policy
          - int
          - 1
          - ``run``
          - 0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY
        * - --gpu_perf_hint
          - int
          - 3
          - ``run``
          - 0:DEFAULT/1:LOW/2:NORMAL/3:HIGH
        * - --gpu_priority_hint
          - int
          - 3
          - ``run``/``benchmark``
          - 0:DEFAULT/1:LOW/2:NORMAL/3:HIGH

Use ``-h`` to get detailed help.

.. code-block:: sh

    python tools/converter.py -h
    python tools/converter.py build -h
    python tools/converter.py run -h

Reduce Library Size
-------------------
* Build for your own usage purpose. Some configuration variables in tools/bazel_build_standalone_lib.sh
  are set to ``true`` by default, you can change them to ``false`` to reduce the library size.
    * **dynamic library**

        - If the models don't need to run on device ``dsp``, change the build option ``--define hexagon=true``
          to ``false``. And the library will be decreased about ``100KB``.

        - Futher more, if only ``cpu`` device needed, change ``--define opencl=true`` to ``false``. This way
          will reduce half of library size to about ``700KB`` for ``armeabi-v7a`` and ``1000KB`` for ``arm64-v8a``

        - About ``300KB`` can be reduced when add ``--config symbol_hidden`` building option. It will change
          the visibility of inner apis in libmace.so and lead to linking error when load model(s) in ``code``
          but no effection for ``file`` mode.

    * **static library**

        - The methods in dynamic library can be useful for static library too. In additional, the static
          library may also contain model graph and model datas if the configs ``model_graph_format`` and
          ``model_data_format`` in deployment file are set to ``code``.

        - It is recommended to use ``version script`` and ``strip`` feature when linking mace static library. The effect is remarkable.

* Remove the unused ops.

Remove the registration of the ops unused for your models in the ``mace/ops/ops_register.cc``,
which will reduce the library size significantly. the final binary just link the registered ops' code.

.. code-block:: cpp

    #include "mace/ops/ops_register.h"

    namespace mace {
    namespace ops {
    // Just leave the ops used in your models

    ...

    }  // namespace ops


    OpRegistry::OpRegistry() {
    // Just leave the ops used in your models

      ...

      ops::RegisterMyCustomOp(this);

      ...

    }

    }  // namespace mace

Reduce Model Size
-------------------
Model file size can be a bottleneck for the deployment of neural networks on mobile devices,
so MACE provides several ways to reduce the model size with no or little performance or accuracy degradation.

**1. Save model weights in half-precision floating point format**

The data type of a regular model is float (32bit). To reduce the model weights size,
half (16bit) can be used to reduce it by half with negligible accuracy degradation.
Therefore, the default storage type for a regular model in MACE is half. However,
if the model is very sensitive to accuracy, storage type can be changed to float.

In the deployment file, ``data_type`` is ``fp16_fp32`` by default and can be changed to ``fp32_fp32``,
for CPU it can also be changed to ``bf16_fp32`` and ``fp16_fp16`` ( ``fp16_fp16`` can only be used on armv8.2 or higher version). 

For CPU, ``fp16_fp32`` means that the weights are saved in half and actual inference is in float; while ``bf16_fp32`` means that the weights are saved in bfloat16 and actual inference is in float,85G
and ``fp16_fp16`` means that the weights are saved in half and actual inference is in half.

For GPU, ``fp16_fp32`` means that the ops in GPU take half as inputs and outputs while kernel execution in float.

**2. Save model weights in quantized fixed point format**

Weights of convolutional (excluding depthwise) and fully connected layers take up a major part of model size.
These weights can be quantized to 8bit to reduce the size to a quarter, whereas the accuracy usually decreases only by 1%-3%.
For example, the top-1 accuracy of MobileNetV1 after quantization of weights is 68.2% on the ImageNet validation set.
``quantize_large_weights`` can be specified as 1 in the deployment file to save these weights in 8bit and actual inference in float.
It can be used for both CPU and GPU.
