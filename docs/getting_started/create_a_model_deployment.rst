Create a model deployment file
==============================

The first step to deploy your models is to create a YAML model deployment
file.

One deployment file describes a case of model deployment,
each file will generate one static library (if more than one ABIs specified,
there will be one static library for each). The deployment file can contain
one or more models, for example, a smart camera application may contain face
recognition, object recognition, and voice recognition models, which can be
defined in one deployment file.


Example
----------
Here is an example deployment file used by an Android demo application.

TODO: change this example file to the demo deployment file
(reuse the same file) and rename to a reasonable name.

.. literalinclude:: models/demo_app_models.yaml
   :language: yaml

Configurations
--------------------

.. list-table::
    :widths: auto
    :header-rows: 1
    :align: left

    * - library_name
      - library name.
    * - target_abis
      - The target ABI to build, can be one or more of 'host', 'armeabi-v7a' or 'arm64-v8a'.
    * - target_socs
      - [optional] build for specified socs if you just want use the model for that socs.
    * - embed_model_data
      - Whether embedding model weights as the code, default to 0.
    * - build_type
      - model build type, can be ['proto', 'code']. 'proto' for converting model to ProtoBuf file and 'code' for converting model to c++ code.
    * - linkshared
      - [optional] Use dynamic linking for libmace library when setting to 1, or static linking when setting to 0, default to 0.
    * - model_name
      - model name, should be unique if there are multiple models.
        **LIMIT: if build_type is code, model_name will used in c++ code so that model_name must fulfill c++ name specification.**
    * - platform
      - The source framework, one of [tensorflow, caffe].
    * - model_file_path
      - The path of the model file, can be local or remote.
    * - model_sha256_checksum
      - The SHA256 checksum of the model file.
    * - weight_file_path
      - [optional] The path of the model weights file, used by Caffe model.
    * - weight_sha256_checksum
      - [optional] The SHA256 checksum of the weight file, used by Caffe model.
    * - subgraphs
      - subgraphs key. **DO NOT EDIT**
    * - input_tensors
      - The input tensor names (tensorflow), top name of inputs' layer (caffe). one or more strings.
    * - output_tensors
      - The output tensor names (tensorflow), top name of outputs' layer (caffe). one or more strings.
    * - input_shapes
      - The shapes of the input tensors, in NHWC order.
    * - output_shapes
      - The shapes of the output tensors, in NHWC order.
    * - input_ranges
      - The numerical range of the input tensors, default [-1, 1]. It is only for test.
    * - validation_inputs_data
      - [optional] Specify Numpy validation inputs. When not provided, [-1, 1] random values will be used.
    * - runtime
      - The running device, one of [cpu, gpu, dsp, cpu_gpu]. cpu_gpu contains CPU and GPU model definition so you can run the model on both CPU and GPU.
    * - data_type
      - [optional] The data type used for specified runtime. [fp16_fp32, fp32_fp32] for GPU, default is fp16_fp32. [fp32] for CPU. [uint8] for DSP.
    * - limit_opencl_kernel_time
      - [optional] Whether splitting the OpenCL kernel within 1 ms to keep UI responsiveness, default to 0.
    * - nnlib_graph_mode
      - [optional] Control the DSP precision and performance, default to 0 usually works for most cases.
    * - obfuscate
      - [optional] Whether to obfuscate the model operator name, default to 0.
    * - winograd
      - [optional] Whether to enable Winograd convolution, **will increase memory consumption**.
