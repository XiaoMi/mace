Advanced usage for CMake users
===============================

This part contains the full usage of MACE.


Deployment file
---------------

There are many advanced options supported.

* **Example**

    Here is an example deployment file with two models.

    .. literalinclude:: models/demo_models_cmake.yml
        :language: yaml


* **Configurations**


.. list-table::
    :header-rows: 1

    * - Options
      - Usage
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
      - [optional] The data type used for specified runtime. [fp16_fp32, fp32_fp32] for GPU, default is fp16_fp32, [fp32] for CPU and [uint8] for DSP.
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

    .. code:: bash

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

  .. code:: bash

    python tools/python/run_model.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml --validate --device_yml=/path/to/devices.yml

There are two steps to do before run:

1. configure login without password

    MACE use ssh to connect embedded device, you should copy your public key to embedded device with the blow command.

    .. code:: bash

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


Model Protection
--------------------------------

Model can be encrypted by obfuscation.

    .. code:: bash

        python tools/python/encrypt.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml

It will override ``mobilenet_v1.pb`` and ``mobilenet_v1.data``. 
If you want to compiled the model into a library, you should use options ``--gencode_model --gencode_param`` to generate model code, i.e.,

    .. code:: bash
    
        python tools/python/encrypt.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml --gencode_model --gencode_param

It will generate model code into ``mace/codegen/models`` and also generate a helper function ``CreateMaceEngineFromCode`` in ``mace/codegen/engine/mace_engine_factory.h`` by which you can create an engine with models built in it.

After that you can rebuild the engine. 
    
    .. code:: bash

        RUNTIME=GPU RUNMODE=code bash tools/cmake/cmake-build-armeabi-v7a.sh

``RUNMODE=code`` means you compile and link model library with MACE engine.

When you test the model in code format, you should specify it in the script as follows.
    
    .. code:: bash

        python tools/python/run_model.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml --gencode_model --gencode_param

Of course you can generate model code only, and use parameter file.

When you need to integrate the libraries into your applications, you can link `libmace_static.a` and `libmodel.a` to your target. These are under the directory:
``build/cmake-build/armeabi-v7a/install/lib/``, the header files you need are under ``build/cmake-build/armeabi-v7a/install/include``.

Refer to \ ``mace/tools/mace_run.cc``\ for full usage. The following list the key steps.

    .. code:: cpp

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


Tuning for specific SoC's GPU
---------------------------------

If you want to use the GPU of a specific device, you can tune the performance for particular devices, which may get 1~10% performance improvement.

You can specify `--tune` option when you want to run and tune the performance at the same time.

    .. code:: bash
    
        python tools/python/run_model.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml --tune

It will generate OpenCL tuned parameter binary file in `build/mobilenet_v1/opencl` directory.
    
    .. code:: bash

        └── mobilenet_v1_tuned_opencl_parameter.MIX2S.sdm845.bin

It specifies your test platform model and SoC. You can use it in production to reduce latency on GPU.

To deploy it, change the names of files generated above for not collision and push them to **your own device's directory**.
Use like the previous procedure, below lists the key steps differently.

    .. code:: cpp

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


Multi Model Support (optional)
--------------------------------

If multiple models are configured in config file. After you test it, it will generate more than one tuned parameter files.
Then you need to merge them together.

    .. code:: bash

        python tools/python/gen_opencl.py

After that, it will generate one set of files into `build/opencl` directory.

    .. code:: bash

        ├── compiled_opencl_kernel.bin
        └── tuned_opencl_parameter.bin


You can also generate code into the engine by specify ``--gencode``, after which you should rebuild the engine.


Validate accuracy of MACE model
-------------------------------

MACE supports **python validation script** as a plugin to test the accuracy, the plugin script could be used for below two purpose.

1. Test the **accuracy(like Top-1)** of MACE model(specifically quantization model) converted from other framework(like tensorflow)
2. Show some real output if you want to see it.

The script define some interfaces like `preprocess` and `postprocess` to deal with input/outut and calculate the accuracy,
you could refer to the `sample code <https://github.com/XiaoMi/mace/tree/master/tools/accuracy_validator.py>`__ for detail.
the sample code show how to calculate the Top-1 accuracy with imagenet validation dataset.


Reduce Library Size
-------------------

Remove the registration of the ops unused for your models in the ``mace/ops/ops_register.cc``,
which will reduce the library size significantly. the final binary just link the registered ops' code.

.. code:: cpp

    #include "mace/ops/ops_register.h"

    namespace mace {
    namespace ops {
    // Just leave the ops used in your models

    ...

    }  // namespace ops


    OpRegistry::OpRegistry() : OpRegistryBase() {
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

The default data type of a regular model is float (32bit). To reduce the model weights size,
half (16bit) can be used to reduce it by half with negligible accuracy degradation.

For CPU, ``data_type`` can be specified as ``fp16_fp32`` in the deployment file to save the weights in half and actual inference in float.

For GPU, ``fp16_fp32`` is default. The ops in GPU take half as inputs and outputs while kernel execution in float.

**2. Save model weights in quantized fixed point format**

Weights of convolutional (excluding depthwise) and fully connected layers take up a major part of model size.
These weights can be quantized to 8bit to reduce the size to a quarter, whereas the accuracy usually decreases only by 1%-3%.
For example, the top-1 accuracy of MobileNetV1 after quantization of weights is 68.2% on the ImageNet validation set.
``quantize_large_weights`` can be specified as 1 in the deployment file to save these weights in 8bit and actual inference in float.
It can be used for both CPU and GPU.
