Basic usage
=============


Build and run an example model
--------------------------------

At first, make sure the environment has been set up correctly already (refer to :doc:`installation`).

The followings are instructions about  how to quickly build and run a provided model in *MACE Model Zoo*.

Here we use the mobilenet-v2 model as an example.

**Commands**

    1. Pull *MACE* project.

    .. code:: sh

        git clone https://github.com/XiaoMi/mace.git
        git fetch --all --tags --prune

        # Checkout the latest tag (i.e. release version)
        tag_name=`git describe --abbrev=0 --tags`
        git checkout tags/${tag_name}

    .. note::

        It's highly recommanded to use a release version instead of master branch.


    2. Pull *MACE Model Zoo* project.

    .. code:: sh

        git clone https://github.com/XiaoMi/mace-models.git


    3. Build MACE.

    .. code:: sh

        cd path/to/mace
        # Build library
        python tools/converter.py build --config=path/to/mace-models/mobilenet-v2/mobilenet-v2.yml


    4. Convert the model to MACE format model.

    .. code:: sh

        cd path/to/mace
        # Build library
        python tools/converter.py build --config=path/to/mace-models/mobilenet-v2/mobilenet-v2.yml


    5. Run the model.

    .. code:: sh

    	# Test model run time
        python tools/converter.py run --config=path/to/mace-models/mobilenet-v2/mobilenet-v2.yml --round=100

    	# Validate the correctness by comparing the results against the
    	# original model and framework, measured with cosine distance for similarity.
    	python tools/converter.py run --config=path/to/mace-models/mobilenet-v2/mobilenet-v2.yml --validate


Build your own model
----------------------------

This part will show you how to use your pre-trained model in MACE.

==================================
1. Prepare your model
==================================

Mace now supports models from Tensorflow and Caffe(more frameworks will be supported).

-  TensorFlow

   Prepare your pre-trained Tensorflow model.pb file.

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


============================================
2. Create a deployment file for your model
============================================

When converting a model or building a library, MACE needs to read a YAML file which is called model deployment file here.

A model deployment file contains all the information of your model(s) and building options. There are several example
deployment files in *MACE Model Zoo* project.

The following shows two basic usage of deployment files for Tensorflow and Caffe models.
Modify one of them and use it for your own case.

-  Tensorflow

   .. literalinclude:: models/demo_app_models_tf.yml
      :language: yaml

-  Caffe

   .. literalinclude:: models/demo_app_models_caffe.yml
      :language: yaml

More details about model deployment file, please refer to :doc:`advanced_usage`.

======================================
3. Convert your model
======================================

When the deployment file is ready for your model, you can use MACE converter tool to convert your model(s).

To convert your pre-trained model to a MACE model, you need to set ``build_type:proto`` in your model deployment file.

And then run this command:

.. code:: bash

    python tools/converter.py convert --config=path/to/your/model_deployment.yml

This command will download or load your pre-trained model and convert it to a MACE model proto file and weights file.
The generated model files will be stored in ``build/${library_name}/model`` folder.

.. warning::

    Please set ``build_type:proto`` in your deployment file before converting.
    The usage of ``build_type:code`` will be demonstrated in :doc:`advanced_usage`.

======================================
4. Build MACE into a library
======================================

MACE can be built into either a static or a shared library (which is
specified by ``linkshared`` in YAML model deployment file).

Use bazel to build MACE source code into a library.

    .. code:: sh

        cd path/to/mace
        # Build library
        bazel build --config=path/to/your/model_deployment_file.yml

The above command will generate library files in the ``build/${library_name}/libs`` folder.

    .. warning::

        1. Please verify the target_abis params in the above command and the deployment file are the same.
        2. If you want to build a library for a specific soc, please refer to :doc:`advanced_usage`.


======================================
5. Run your model
======================================

With the converted model, *.so or *.a library and header files, you can use the following commands to run and validate your model.

* **run**

    run the model.

    .. code:: sh

    	# Test model run time
        python tools/converter.py run --config=path/to/your/model_deployment_file.yml --round=100

    	# Validate the correctness by comparing the results against the
    	# original model and framework, measured with cosine distance for similarity.
    	python tools/converter.py run --config=path/to/your/model_deployment_file.yml --validate

* **benchmark**

    benchmark and profile the model.

    .. code:: sh

        # Benchmark model, get detailed statistics of each Op.
        python tools/converter.py benchmark --config=path/to/your/model_deployment_file.yml


========================================================
6. Deploy your model into applications
========================================================

In the converting and building steps, you've got the static/shared library, model files and
header files. All of these generated files have been packaged into
``build/${library_name}/libmace_${library_name}.tar.gz`` when building.

``${library_name}`` is the name you defined in the first line of your deployment YAML file.

-  The generated ``static`` library files are organized as follows,

.. code::

      build/
      └── mobilenet-v2
          ├── include
          │   └── mace
          │       └── public
          │           ├── mace.h
          │           └── mace_runtime.h
          ├── libmace_mobilenet-v2.tar.gz
          ├── lib
          │   ├── arm64-v8a
          │   │   └── libmace_mobilenet-v2.MI6.msm8998.a
          │   └── armeabi-v7a
          │       └── libmace_mobilenet-v2.MI6.msm8998.a
          ├── model
          │   ├── mobilenet_v2.data
          │   └── mobilenet_v2.pb
          └── opencl
              ├── arm64-v8a
              │   └── mobilenet-v2_compiled_opencl_kernel.MI6.msm8998.bin
              └── armeabi-v7a
                  └── mobilenet-v2_compiled_opencl_kernel.MI6.msm8998.bin

-  The generated ``shared`` library files are organized as follows,

.. code::

      build
      └── mobilenet-v2
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
              │   └── mobilenet-v2_compiled_opencl_kernel.MI6.msm8998.bin
              └── armeabi-v7a
                  └── mobilenet-v2_compiled_opencl_kernel.MI6.msm8998.bin


Unpack the generated libmace_${library_name}.tar.gz file and copy all of the uncompressed files into your project.

Please refer to \ ``mace/examples/example.cc``\ for full usage. The following list the key steps.

.. code:: cpp

    // Include the headers
    #include "mace/public/mace.h"
    #include "mace/public/mace_runtime.h"

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

More details are in :doc:`advanced_usage`.