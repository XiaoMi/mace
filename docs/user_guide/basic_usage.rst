Basic usage
=============


Build and run an example model
--------------------------------

Make sure the environment has been set up correctly already.(refer to `installation`)

Pull the mace model zoo project.

.. code:: sh

    git clone https://github.com/XiaoMi/mace-models.git

Here we use the provided mobilenet-v2 model in mace model zoo as an example.
Plug an android phone into your pc and enable Developer Mode of the phone.

.. code:: sh

    cd /path/to/mace
    python tools/converter.py build --config=/path/to/mace-models/mobilenet-v2/mobilenet-v2.yml

Validate and benchmark the model.

.. code:: sh

    # Validate the model.
    python tools/converter.py run --config=/path/to/mace-models/mobilenet-v2/mobilenet-v2.yml --validate
    # Benchmark
    python tools/converter.py benchmark --config=/path/to/mace-models/mobilenet-v2/mobilenet-v2.yml

.. note::

     1. If you want to build and run the model on pc, just use the mobilenet-v2-host.yml file instead.


Build your own model
----------------------------
==================================
1. Prepare your model
==================================

Mace now supports models from tensorflow and caffe.

-  TensorFlow

   Prepare your tensorflow model.pb file.

   Use `Graph Transform Tool <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md>`__
   to optimize you model for inference.
   This tool will improve the efficiency of inference by making several optimizations like operations
   folding, redundant node removal etc. We strongly recommend to use it before building.

   The following command shows how to use it for CPU/GPU,

   .. code:: bash

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

-  Caffe

   MACE converter only supports Caffe 1.0+, you need to upgrade
   your models with Caffe built-in tool if necessary,

   .. code:: bash

       # Upgrade prototxt
       $CAFFE_ROOT/build/tools/upgrade_net_proto_text MODEL.prototxt MODEL.new.prototxt

       # Upgrade caffemodel
       $CAFFE_ROOT/build/tools/upgrade_net_proto_binary MODEL.caffemodel MODEL.new.caffemodel

============================================
2. Create a deployment file for your model
============================================

The followings are basic usage example deployment files for Tensorflow and Caffe models.
Modify one of them for your own case.

-  Tensorflow

   .. literalinclude:: models/demo_app_models_tf.yml
      :language: yaml

-  Caffe

   .. literalinclude:: models/demo_app_models_caffe.yml
      :language: yaml

More details about model deployment file, refer to `advanced_usage`.


======================================
3. Build a library for your model
======================================

MACE provides a python tool (``tools/converter.py``) for
model conversion, compiling, test run, benchmark and correctness validation.

MACE can build either static or shared link library (which is
specified by ``linkshared`` in YAML model deployment file).

**Commands**

    * **build**

        build library.

    .. code:: sh

        cd path/to/mace
        # Build library
        python tools/converter.py build --config=path/to/your/model_deployment_file.yml

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

    .. warning::

        1. Plug an android phone into your pc and enable Developer Mode before building.
        2. Please ``build`` your model before ``run`` or ``benchmark`` it.


============================================
4. Deploy generated library in your project
============================================

``build`` command will generate the static/shared library, model files and
header files. All of these files have been packaged into
``path/to/mace/build/${library_name}/libmace_${library_name}.tar.gz``.

``${library_name}`` is the name you defined in the first line of your demployment yaml file.

-  The generated ``static`` library files are organized as follows,

.. code::

      build/
      └── mobilenet-v2-gpu
          ├── include
          │   └── mace
          │       └── public
          │           ├── mace.h
          │           └── mace_runtime.h
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

-  The generated ``shared`` library files are organized as follows,

.. code::

      build
      └── mobilenet-v2-gpu
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
              │   └── mobilenet-v2-gpu_compiled_opencl_kernel.MI6.msm8998.bin
              └── armeabi-v7a
                  └── mobilenet-v2-gpu_compiled_opencl_kernel.MI6.msm8998.bin

.. note::

    1. ``${MODEL_TAG}.pb`` file will be generated only when ``build_type`` is ``proto``.
    2. ``${library_name}_compiled_opencl_kernel.${device_name}.${soc}.bin`` will
       be generated only when ``target_socs`` and ``gpu`` runtime are specified.
    3. Generated shared library depends on ``libgnustl_shared.so``.

.. warning::

    ``${library_name}_compiled_opencl_kernel.${device_name}.${soc}.bin`` depends
    on the OpenCL version of the device, you should maintan the compatibility or
    configure compiling cache store with ``ConfigKVStorageFactory``.


Unpack the generated libmace_${library_name}.tar.gz file and copy all of the uncompressed files into your project.
Please refer to \ ``mace/examples/example.cc``\ for full usage. The following lists the key steps.

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

    // 1. Set compiled OpenCL kernel cache to reduce the
    // initialization time.
    const std::string file_path ="path/to/opencl_cache_file";
    std::shared_ptr<KVStorageFactory> storage_factory(
        new FileStorageFactory(file_path));
    ConfigKVStorageFactory(storage_factory);

    // 2. Declare the device type (must be same with ``runtime`` in deployment file)
    DeviceType device_type = DeviceType::GPU;

    // 3. Define the input and output tensor names.
    std::vector<std::string> input_names = {...};
    std::vector<std::string> output_names = {...};

    // 4. Create MaceEngine instance
    std::shared_ptr<mace::MaceEngine> engine;
    MaceStatus create_engine_status;
    // If the build_type is code, create Engine from compiled code
    create_engine_status =
        CreateMaceEngineFromCode(model_name.c_str(),
                                 nullptr,
                                 input_names,
                                 output_names,
                                 device_type,
                                 &engine);
    // If the build_type is proto, Create Engine from model file
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
    engine->Run(inputs, &outputs);

More details in `advanced_usage`.