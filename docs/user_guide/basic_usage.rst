Basic usage for Bazel users
============================


Build and run an example model
-------------------------------

At first, make sure the environment has been set up correctly already (refer to :doc:`../installation/env_requirement.rst`).

The followings are instructions about how to quickly build and run a provided model in
`MACE Model Zoo <https://github.com/XiaoMi/mace-models>`__.

Here we use the mobilenet-v2 model as an example.

**Commands**

    1. Pull `MACE <https://github.com/XiaoMi/mace>`__ project.

    .. code-block:: sh

        git clone https://github.com/XiaoMi/mace.git
        cd mace/
        git fetch --all --tags --prune

        # Checkout the latest tag (i.e. release version)
        tag_name=`git describe --abbrev=0 --tags`
        git checkout tags/${tag_name}

    .. note::

        It's highly recommended to use a release version instead of master branch.


    2. Pull `MACE Model Zoo <https://github.com/XiaoMi/mace-models>`__ project.

    .. code-block:: sh

        git clone https://github.com/XiaoMi/mace-models.git


    3. Build a generic MACE library.

    .. code-block:: sh

        cd path/to/mace
        # Build library
        # output lib path: build/lib
        bash tools/bazel_build_standalone_lib.sh [-abi=abi][-runtimes=rt1,rt2,...][-quantize][-static][-rpcmem]

    .. note::

        - This step can be skipped if you just want to run a model using ``tools/converter.py``, such as commands in step 5.
        - Use the `-abi` parameter to specify the ABI. Supported ABIs are armeabi-v7a, arm64-v8a, arm_linux_gnueabihf, aarch64_linux_gnu and host (for host machine, linux-x86-64). The default ABI is arm64-v8a.
        - For each ABI, several runtimes can be chosen by specifying the `-runtimes` parameter. Supported runtimes are CPU, GPU, DSP and APU. By default, the library is built to run on CPU.
        - Omit the `-static` option if a shared library is desired instead of a static one. By default, a shared library is built.
        - Omit the `-rpcmem` option if your target device chipset is not manufactured by Qualcomm.
        - See 'bash tools/bazel_build_standalone_lib.sh -help' for detailed information.
        - DO respect the hyphens ('-') and the underscores ('_') in the ABI.


    4. Convert the pre-trained mobilenet-v2 model to MACE format model.

    .. code-block:: sh

        cd path/to/mace
        # Build library
        python tools/converter.py convert --config=/path/to/mace-models/mobilenet-v2/mobilenet-v2.yml


    5. Run the model.

    .. note::

        If you want to run on phone, please plug in at least one phone.
        Or if you want to run on embedded device, please give a :doc:`advanced_usage`.

    .. code-block:: sh

        # Run
        python tools/converter.py run --config=/path/to/mace-models/mobilenet-v2/mobilenet-v2.yml

    	# Test model run time
        python tools/converter.py run --config=/path/to/mace-models/mobilenet-v2/mobilenet-v2.yml --round=100

    	# Validate the correctness by comparing the results against the
    	# original model and framework, measured with cosine distance for similarity.
    	python tools/converter.py run --config=/path/to/mace-models/mobilenet-v2/mobilenet-v2.yml --validate


Build your own model
---------------------

This part will show you how to use your own pre-trained model in MACE.

======================
1. Prepare your model
======================

MACE now supports models from TensorFlow and Caffe (more frameworks will be supported).

-  TensorFlow

   Prepare your pre-trained TensorFlow model.pb file.

-  Caffe

   Caffe 1.0+ models are supported in MACE converter tool.

   If your model is from lower version Caffe, you need to upgrade it by using the Caffe built-in tool before converting.

   .. code-block:: bash

       # Upgrade prototxt
       $CAFFE_ROOT/build/tools/upgrade_net_proto_text MODEL.prototxt MODEL.new.prototxt

       # Upgrade caffemodel
       $CAFFE_ROOT/build/tools/upgrade_net_proto_binary MODEL.caffemodel MODEL.new.caffemodel

-  ONNX

   Prepare your ONNX model.onnx file.

   Use `ONNX Optimizer Tool <https://github.com/XiaoMi/mace/tree/master/tools/onnx_optimizer.py>`__ to optimize your model for inference.
   This tool will improve the efficiency of inference like the `Graph Transform Tool <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md>`__
   in TensorFlow.

   .. code-block:: bash

       # Optimize your model
       $python MACE_ROOT/tools/onnx_optimizer.py model.onnx model_opt.onnx


===========================================
2. Create a deployment file for your model
===========================================

When converting a model or building a library, MACE needs to read a YAML file which is called model deployment file here.

A model deployment file contains all the information of your model(s) and building options. There are several example
deployment files in *MACE Model Zoo* project.

The following shows two basic usage of deployment files for TensorFlow and Caffe models.
Modify one of them and use it for your own case.

-  TensorFlow

   .. literalinclude:: models/demo_models_tf.yml
      :language: yaml

-  Caffe

   .. literalinclude:: models/demo_models_caffe.yml
      :language: yaml

-  ONNX

   .. literalinclude:: models/demo_models_onnx.yml
      :language: yaml


More details about model deployment file are in :doc:`advanced_usage`.

======================
3. Convert your model
======================

When the deployment file is ready, you can use MACE converter tool to convert your model(s).

.. code-block:: bash

    python tools/converter.py convert --config=/path/to/your/model_deployment_file.yml

This command will download or load your pre-trained model and convert it to a MACE model proto file and weights data file.
The generated model files will be stored in ``build/${library_name}/model`` folder.

.. warning::

    Please set ``model_graph_format: file`` and ``model_data_format: file`` in your deployment file before converting.
    The usage of ``model_graph_format: code`` will be demonstrated in :doc:`advanced_usage`.

=============================
4. Build MACE into a library
=============================
You could Download the prebuilt MACE Library from `Github MACE release page <https://github.com/XiaoMi/mace/releases>`__.

Or use bazel to build MACE source code into a library.

    .. code-block:: sh

        cd path/to/mace
        # Build library
        # output lib path: build/lib
        bash tools/bazel_build_standalone_lib.sh [-abi=abi][-runtimes=rt1,rt2,...][-static][-rpcmem]

The above command will generate static library ``build/lib/libmace.a`` dynamic library ``build/lib/libmace.so``.

    .. warning::

        Please verify that the -abi param in the above command is the same as the target_abi param in your deployment file.

==================
5. Run your model
==================

With the converted model, the static or shared library and header files, you can use the following commands
to run and validate your model.

    .. warning::

        If you want to run on device/phone, please plug in at least one device/phone.

* **run**

    run the model.

    .. code-block:: sh

    	# Test model run time
        python tools/converter.py run --config=/path/to/your/model_deployment_file.yml --round=100

        # Validate the correctness by comparing the results against the
    	# original model and framework, measured with cosine distance for similarity.
    	python tools/converter.py run --config=/path/to/your/model_deployment_file.yml --validate

        # If you want to run model on specified arm linux device, you should put device config file in the working directory or run with flag `--device_yml`
        python tools/converter.py run --config=/path/to/your/model_deployment_file.yml --device_yml=/path/to/devices.yml


* **benchmark**

    benchmark and profile the model. the details are in :doc:`benchmark`.

    .. code-block:: sh

        # Benchmark model, get detailed statistics of each Op.
        python tools/converter.py run --config=/path/to/your/model_deployment_file.yml --benchmark


=======================================
6. Deploy your model into applications
=======================================

You could run model on CPU, GPU and DSP (based on the `runtime` in your model deployment file).
However, there are some differences in different devices.

* **CPU**

    Almost all of mobile SoCs use ARM-based CPU architecture, so your model could run on different SoCs in theory.

* **GPU**

    Although most GPUs use OpenCL standard, but there are some SoCs not fully complying with the standard,
    or the GPU is too low-level to use. So you should have some fallback strategies when the GPU run failed.

* **DSP**

    MACE only supports Qualcomm DSP. And you need to push the hexagon nn library to the device.

    .. code-block:: sh

        # For Android device
        adb root; adb remount
        adb push third_party/nnlib/v6x/libhexagon_nn_skel.so /system/vendor/lib/rfsa/adsp/

In the converting and building steps, you've got the static/shared library, model files and
header files.


``${library_name}`` is the name you defined in the first line of your deployment YAML file.

.. note::

    When linking generated ``libmace.a`` into shared library,
    `version script <ftp://ftp.gnu.org/old-gnu/Manuals/ld-2.9.1/html_node/ld_25.html>`__
    is helpful for reducing a specified set of symbols to local scope.

-  The generated ``static`` library files are organized as follows,

.. code-block:: none

    build
    ├── include
    │   └── mace
    │       └── public
    │           └── mace.h
    ├── lib
    │   ├── libmace.a	(for static library)
    │   ├── libmace.so	(for shared library)
    │   └── libhexagon_controller.so	(for DSP runtime)
    └── mobilenet-v1
        ├── model
        │   ├── mobilenet_v1.data
        │   └── mobilenet_v1.pb
        └── _tmp
            └── arm64-v8a
                └── mace_run_static


Please refer to \ ``mace/tools/mace_run.cc``\ for full usage. The following list the key steps.

.. code-block:: cpp

    // Include the headers
    #include "mace/public/mace.h"

    // 0. Declare the device type (must be same with ``runtime`` in configuration file)
    DeviceType device_type = DeviceType::GPU;

    // 1. configuration
    MaceStatus status;
    MaceEngineConfig config(device_type);
    std::shared_ptr<GPUContext> gpu_context;
    // Set the path to store compiled OpenCL kernel binaries.
    // please make sure your application have read/write rights of the directory.
    // this is used to reduce the initialization time since the compiling is too slow.
    // It's suggested to set this even when pre-compiled OpenCL program file is provided
    // because the OpenCL version upgrade may also leads to kernel recompilations.
    const std::string storage_path ="path/to/storage";
    gpu_context = GPUContextBuilder()
        .SetStoragePath(storage_path)
        .Finalize();
    config.SetGPUContext(gpu_context);
    config.SetGPUHints(
        static_cast<GPUPerfHint>(GPUPerfHint::PERF_NORMAL),
        static_cast<GPUPriorityHint>(GPUPriorityHint::PRIORITY_LOW));

    // 2. Define the input and output tensor names.
    std::vector<std::string> input_names = {...};
    std::vector<std::string> output_names = {...};

    // 3. Create MaceEngine instance
    std::shared_ptr<mace::MaceEngine> engine;
    MaceStatus create_engine_status;

    // Create Engine from model file
    create_engine_status =
        CreateMaceEngineFromProto(model_graph_proto,
                                  model_graph_proto_size,
                                  model_weights_data,
                                  model_weights_data_size,
                                  input_names,
                                  output_names,
                                  device_type,
                                  &engine);
    if (create_engine_status != MaceStatus::MACE_SUCCESS) {
      // fall back to other strategy.
    }

    // 4. Create Input and Output tensor buffers
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

    // 5. Run the model
    MaceStatus status = engine.Run(inputs, &outputs);

More details are in :doc:`advanced_usage`.

