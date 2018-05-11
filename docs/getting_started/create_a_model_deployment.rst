Create a model deployment file
==============================

The first step to deploy you models is to create a YAML model deployment
file.

One deployment file describes a case of model deployment,
each file will generate one static library (if more than one ABIs specified,
there will be one static library for each). The deployment file can contains
one or more models, for example, a smart camera application may contains face
recognition, object recognition, and voice recognition models, which can be
defined in one deployment file),


Example
----------
Here is an deployment file example used by Android demo application.

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

    * - Configuration key
      - Description
    * - target_abis
      - The target ABI to build, can be one or more of 'host', 'armeabi-v7a' or 'arm64-v8a'
    * - embed_model_data
      - Whether embedding model weights as the code, default to 1
    * - platform
      - The source framework, tensorflow or caffe
    * - model_file_path
      - The path of the model file, can be local or remote
    * - weight_file_path
      - The path of the model weights file, used by Caffe model
    * - model_sha256_checksum
      - The SHA256 checksum of the model file
    * - weight_sha256_checksum
      - The SHA256 checksum of the weight file, used by Caffe model
    * - input_nodes
      - The input node names, one or more strings
    * - output_nodes
      - The output node names, one or more strings
    * - input_shapes
      - The shapes of the input nodes, in NHWC order
    * - output_shapes
      - The shapes of the output nodes, in NHWC order
    * - runtime
      - The running device, one of CPU, GPU or DSP
    * - limit_opencl_kernel_time
      - Whether splitting the OpenCL kernel within 1 ms to keep UI responsiveness, default to 0
    * - dsp_mode
      - Control the DSP precision and performance, default to 0 usually works for most cases
    * - obfuscate
      - Whether to obfuscate the model operator name, default to 0
    * - fast_conv
      - Whether to enable Winograd convolution, **will increase memory consumption**
    * - input_files
      - Specify Numpy validation inputs. When not provided, [-1, 1] random values will be used
