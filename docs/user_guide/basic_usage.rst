Basic usage
============


Build and run an example model
-------------------------------

At first, make sure the environment has been set up correctly already (refer to :doc:`../installation/env_requirement`).

The followings are instructions about how to quickly build and run a provided model in
`MACE Model Zoo <https://github.com/XiaoMi/mace-models>`__.

Here we use the mobilenet-v2 model as an example.

**Commands**

    1. Pull `MACE <https://github.com/XiaoMi/mace>`__ project.

    .. code:: sh

        git clone https://github.com/XiaoMi/mace.git
        cd mace/
        git fetch --all --tags --prune

        # Checkout the latest tag (i.e. release version)
        tag_name=`git describe --abbrev=0 --tags`
        git checkout tags/${tag_name}

    .. note::

        It's highly recommanded to use a release version instead of master branch.


    2. Pull `MACE Model Zoo <https://github.com/XiaoMi/mace-models>`__ project.

    .. code:: sh

        git clone https://github.com/XiaoMi/mace-models.git


    3. Build a generic MACE library.

    .. code:: sh

        cd path/to/mace
        # Build library
        # output lib path: builds/lib
        bash tools/build-standalone-lib.sh


    .. note::

        - Libraries in ``builds/lib/armeabi-v7a/cpu_gpu/`` means it can run on ``cpu`` or ``gpu`` devices.

        - The results in ``builds/lib/armeabi-v7a/cpu_gpu_dsp/`` need HVX supported.


    4. Convert the pre-trained mobilenet-v2 model to MACE format model.

    .. code:: sh

        cd path/to/mace
        # Build library
        python tools/converter.py convert --config=/path/to/mace-models/mobilenet-v2/mobilenet-v2.yml


    5. Run the model.

    .. note::

        If you want to run on device/phone, please plug in at least one device/phone.

    .. code:: sh

        # Run example
        python tools/converter.py run --config=/path/to/mace-models/mobilenet-v2/mobilenet-v2.yml --example

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

   Use `Graph Transform Tool <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md>`__
   to optimize your model for inference.
   This tool will improve the efficiency of inference by making several optimizations like operators
   folding, redundant node removal etc. We strongly recommend MACE users to use it before building.

   Usage for CPU/GPU,

   .. code:: bash

       # CPU/GPU:
       ./transform_graph \
           --in_graph=/path/to/your/tf_model.pb \
           --out_graph=/path/to/your/output/tf_model_opt.pb \
           --inputs='input node name' \
           --outputs='output node name' \
           --transforms='strip_unused_nodes(type=float, shape="1,64,64,3")
               strip_unused_nodes(type=float, shape="1,64,64,3")
               remove_nodes(op=Identity, op=CheckNumerics)
               fold_constants(ignore_errors=true)
               flatten_atrous_conv
               fold_batch_norms
               fold_old_batch_norms
               remove_control_dependencies
               strip_unused_nodes
               sort_by_execution_order'

	Usage for DSP,

   .. code:: bash

       # DSP:
       ./transform_graph \
           --in_graph=/path/to/your/tf_model.pb \
           --out_graph=/path/to/your/output/tf_model_opt.pb \
           --inputs='input node name' \
           --outputs='output node name' \
           --transforms='strip_unused_nodes(type=float, shape="1,64,64,3")
               strip_unused_nodes(type=float, shape="1,64,64,3")
               remove_nodes(op=Identity, op=CheckNumerics)
               fold_constants(ignore_errors=true)
               fold_batch_norms
               fold_old_batch_norms
               backport_concatv2
               quantize_weights(minimum_size=2)
               quantize_nodes
               remove_control_dependencies
               strip_unused_nodes
               sort_by_execution_order'

-  Caffe

   Caffe 1.0+ models are supported in MACE converter tool.

   If your model is from lower version Caffe, you need to upgrade it by using the Caffe built-in tool before converting.

   .. code:: bash

       # Upgrade prototxt
       $CAFFE_ROOT/build/tools/upgrade_net_proto_text MODEL.prototxt MODEL.new.prototxt

       # Upgrade caffemodel
       $CAFFE_ROOT/build/tools/upgrade_net_proto_binary MODEL.caffemodel MODEL.new.caffemodel


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

More details about model deployment file are in :doc:`advanced_usage`.

======================
3. Convert your model
======================

When the deployment file is ready, you can use MACE converter tool to convert your model(s).

.. code:: bash

    python tools/converter.py convert --config=/path/to/your/model_deployment_file.yml

This command will download or load your pre-trained model and convert it to a MACE model proto file and weights data file.
The generated model files will be stored in ``builds/${library_name}/model`` folder.

.. warning::

    Please set ``model_graph_format: file`` and ``model_data_format: file`` in your deployment file before converting.
    The usage of ``model_graph_format: code`` will be demonstrated in :doc:`advanced_usage`.

=============================
4. Build MACE into a library
=============================
You could Download the prebuilt MACE Library from `Github MACE release page <https://github.com/XiaoMi/mace/releases>`__.

Or use bazel to build MACE source code into a library.

    .. code:: sh

        cd path/to/mace
        # Build library
        # output lib path: builds/lib
        bash tools/build-standalone-lib.sh

The above command will generate dynamic library ``builds/lib/${ABI}/${DEVICES}/libmace.so`` and static library ``builds/lib/${ABI}/${DEVICES}/libmace.a``.

    .. warning::

        Please verify that the target_abis param in the above command and your deployment file are the same.


==================
5. Run your model
==================

With the converted model, the static or shared library and header files, you can use the following commands
to run and validate your model.

    .. warning::

        If you want to run on device/phone, please plug in at least one device/phone.

* **run**

    run the model.

    .. code:: sh

    	# Test model run time
        python tools/converter.py run --config=/path/to/your/model_deployment_file.yml --round=100

    	# Validate the correctness by comparing the results against the
    	# original model and framework, measured with cosine distance for similarity.
    	python tools/converter.py run --config=/path/to/your/model_deployment_file.yml --validate

* **benchmark**

    benchmark and profile the model.

    .. code:: sh

        # Benchmark model, get detailed statistics of each Op.
        python tools/converter.py benchmark --config=/path/to/your/model_deployment_file.yml


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

    MACE only support Qualcomm DSP.

In the converting and building steps, you've got the static/shared library, model files and
header files.


``${library_name}`` is the name you defined in the first line of your deployment YAML file.

.. note::

    When linking generated ``libmace.a`` into shared library,
    `version script <ftp://ftp.gnu.org/old-gnu/Manuals/ld-2.9.1/html_node/ld_25.html>`__
    is helpful for reducing a specified set of symbols to local scope.

-  The generated ``static`` library files are organized as follows,

.. code::

    builds
    ├── include
    │   └── mace
    │       └── public
    │           ├── mace.h
    │           └── mace_runtime.h
    ├── lib
    │   ├── arm64-v8a
    │   │   └── cpu_gpu
    │   │       ├── libmace.a
    │   │       └── libmace.so
    │   ├── armeabi-v7a
    │   │   ├── cpu_gpu
    │   │   │   ├── libmace.a
    │   │   │   └── libmace.so
    │   │   └── cpu_gpu_dsp
    │   │       ├── libhexagon_controller.so
    │   │       ├── libmace.a
    │   │       └── libmace.so
    │   └── linux-x86-64
    │       ├── libmace.a
    │       └── libmace.so
    └── mobilenet-v1
        ├── model
        │   ├── mobilenet_v1.data
        │   └── mobilenet_v1.pb
        └── _tmp
            └── arm64-v8a
                └── mace_run_static


Please refer to \ ``mace/examples/example.cc``\ for full usage. The following list the key steps.

.. code:: cpp

    // Include the headers
    #include "mace/public/mace.h"
    #include "mace/public/mace_runtime.h"

    // 0. Set compiled OpenCL kernel cache, this is used to reduce the
    // initialization time since the compiling is too slow. It's suggested
    // to set this even when pre-compiled OpenCL program file is provided
    // because the OpenCL version upgrade may also leads to kernel
    // recompilations.
    const std::string file_path ="path/to/opencl_cache_file";
    std::shared_ptr<KVStorageFactory> storage_factory(
        new FileStorageFactory(file_path));
    ConfigKVStorageFactory(storage_factory);

    // 1. Declare the device type (must be same with ``runtime`` in configuration file)
    DeviceType device_type = DeviceType::GPU;

    // 2. Define the input and output tensor names.
    std::vector<std::string> input_names = {...};
    std::vector<std::string> output_names = {...};

    // 3. Create MaceEngine instance
    std::shared_ptr<mace::MaceEngine> engine;
    MaceStatus create_engine_status;

    // Create Engine from model file
    create_engine_status =
        CreateMaceEngineFromProto(model_pb_data,
                                  model_data_file.c_str(),
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
