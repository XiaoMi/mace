Create a model deployment
=========================

Each YAML deployment script describes a case of deployments (for example,
a smart camera application may contains face recognition, object recognition,
and voice recognition models, which can be defined in one deployment file),
which will generate one static library (if more than one ABIs specified,
there will be one static library for each). Each YAML scripts can contains one
or more models.


Model deployment file example
-------------------------------
TODO: change to a link to a standalone file with comments.

.. code:: yaml

    # 配置文件名会被用作生成库的名称：libmace-${filename}.a
    target_abis: [armeabi-v7a, arm64-v8a]
    # 具体机型的soc编号，可以使用`adb shell getprop | grep ro.board.platform | cut -d [ -f3 | cut -d ] -f1`获取
    target_socs: [msm8998]
    embed_model_data: 1
    vlog_level: 0
    models: # 一个配置文件可以包含多个模型的配置信息，最终生成的库中包含多个模型
      first_net: # 模型的标签，在调度模型的时候，会用这个变量
        platform: tensorflow
        model_file_path: path/to/model64.pb # also support http:// and https://
        model_sha256_checksum: 7f7462333406e7dea87222737590ebb7d94490194d2f21a7d72bafa87e64e9f9
        input_nodes: input_node
        output_nodes: output_node
        input_shapes: 1,64,64,3
        output_shapes: 1,64,64,2
        runtime: gpu
        limit_opencl_kernel_time: 0
        dsp_mode: 0
        obfuscate: 1
        fast_conv: 0
        input_files:
          - path/to/input_files # support http://
      second_net:
        platform: caffe
        model_file_path: path/to/model.prototxt
        weight_file_path: path/to/weight.caffemodel
        model_sha256_checksum: 05d92625809dc9edd6484882335c48c043397aed450a168d75eb8b538e86881a
        weight_sha256_checksum: 05d92625809dc9edd6484882335c48c043397aed450a168d75eb8b538e86881a
        input_nodes:
          - input_node0
          - input_node1
        output_nodes:
          - output_node0
          - output_node1
        input_shapes:
          - 1,256,256,3
          - 1,128,128,3
        output_shapes:
          - 1,256,256,2
          - 1,1,1,2
        runtime: cpu
        limit_opencl_kernel_time: 1
        dsp_mode: 0
        obfuscate: 1
        fast_conv: 0
        input_files:
          - path/to/input_files # support http://

Configurations
--------------------

+--------------------------+----------------------------------------------------------------------------------------+
| Configuration key        | Description                                                                            |
+==========================+========================================================================================+
| target_abis              | The target ABI to build, can be one or more of 'host', 'armeabi-v7a' or 'arm64-v8a'    |
+--------------------------+----------------------------------------------------------------------------------------+
| embed_model_data         | Whether embedding model weights as the code, default to 1                              |
+--------------------------+----------------------------------------------------------------------------------------+
| platform                 | The source framework, tensorflow or caffe                                              |
+--------------------------+----------------------------------------------------------------------------------------+
| model_file_path          | The path of the model file, can be local or remote                                     |
+--------------------------+----------------------------------------------------------------------------------------+
| weight_file_path         | The path of the model weights file, used by Caffe model                                |
+--------------------------+----------------------------------------------------------------------------------------+
| model_sha256_checksum    | The SHA256 checksum of the model file                                                  |
+--------------------------+----------------------------------------------------------------------------------------+
| weight_sha256_checksum   | The SHA256 checksum of the weight file, used by Caffe model                            |
+--------------------------+----------------------------------------------------------------------------------------+
| input_nodes              | The input node names, one or more strings                                              |
+--------------------------+----------------------------------------------------------------------------------------+
| output_nodes             | The output node names, one or more strings                                             |
+--------------------------+----------------------------------------------------------------------------------------+
| input_shapes             | The shapes of the input nodes, in NHWC order                                           |
+--------------------------+----------------------------------------------------------------------------------------+
| output_shapes            | The shapes of the output nodes, in NHWC order                                          |
+--------------------------+----------------------------------------------------------------------------------------+
| runtime                  | The running device, one of CPU, GPU or DSP                                             |
+--------------------------+----------------------------------------------------------------------------------------+
| limit_opencl_kernel_time | Whether splitting the OpenCL kernel within 1 ms to keep UI responsiveness, default to 0|
+--------------------------+----------------------------------------------------------------------------------------+
| dsp_mode                 | Control the DSP precision and performance, default to 0 usually works for most cases   |
+--------------------------+----------------------------------------------------------------------------------------+
| obfuscate                | Whether to obfuscate the model operator name, default to 0                             |
+--------------------------+----------------------------------------------------------------------------------------+
| fast_conv                | Whether to enable Winograd convolution, **will increase memory consumption**           |
+--------------------------+----------------------------------------------------------------------------------------+
| input_files              | Specify Numpy validation inputs. When not provided, [-1, 1] random values will be used |
+--------------------------+----------------------------------------------------------------------------------------+
