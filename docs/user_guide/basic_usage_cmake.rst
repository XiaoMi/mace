Basic usage for CMake users
=============================

First of all, make sure the environment has been set up correctly already (refer to :doc:`../installation/env_requirement`).

Clear Workspace
-------------------------------

Before you do anything, clear the workspace used by build and test process.

    .. code:: sh

        tools/clear_workspace.sh


Build Engine
-------------------------------

Please make sure you have CMake installed.

    .. code:: sh

        RUNTIME=GPU bash tools/cmake/cmake-build-armeabi-v7a.sh

which generate libraries in ``build/cmake-build/armeabi-v7a``, you can use either static libraries or the ``libmace.so`` shared library.

You can also build for other target abis: ``arm64-v8a``, ``arm-linux-gnueabihf``, ``aarch64-linux-gnu``, ``host``;
and runtime: ``GPU``, ``HEXAGON``, ``HTA``, ``APU``.


Model Conversion
-------------------------------

When you have prepared your model, the first thing to do is write a model config in YAML format.

    .. code:: yaml

        models:
          mobilenet_v1:
            platform: tensorflow
            model_file_path: https://cnbj1.fds.api.xiaomi.com/mace/miai-models/mobilenet-v1/mobilenet-v1-1.0.pb
            model_sha256_checksum: 71b10f540ece33c49a7b51f5d4095fc9bd78ce46ebf0300487b2ee23d71294e6
            subgraphs:
              - input_tensors:
                  - input
                input_shapes:
                  - 1,224,224,3
                output_tensors:
                  - MobilenetV1/Predictions/Reshape_1
                output_shapes:
                  - 1,1001
            runtime: gpu



The following steps generate output to ``build`` directory which is the default build and test workspace.
Suppose you have the model config in ``../mace-models/mobilenet-v1/mobilenet-v1.yml``. Then run

    .. code:: yaml

        python tools/python/convert.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml

which generate 4 files in ``build/mobilenet_v1/model/``

    .. code:: sh

        ├── mobilenet_v1.pb                (model file)
        ├── mobilenet_v1.data              (param file)
        ├── mobilenet_v1_index.html        (visualization page, you can open it in browser)
        └── mobilenet_v1.pb_txt            (model text file, which can be for debug use)


MACE also supports other platform: ``caffe``, ``onnx``.
Beyond GPU, users can specify ``cpu``, ``dsp`` to run on other target devices.


Model Test and Benchmark
-------------------------------

We provide simple tools to test and benchmark your model.

After model is converted, simply run

    .. code:: sh

        python tools/python/run_model.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml --validate

Or benchmark the model

    .. code:: sh

        python tools/python/run_model.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml --benchmark



It will test your model on the device configured in the model config (``runtime``).
You can also test on other device by specify ``--runtime=cpu (dsp/hta/apu)`` when you run test if you previously build engine for the device.
The log will be shown if ``--vlog_level=2`` is specified.



Deploy your model into applications
--------------------------------------

Please refer to \ ``mace/tools/mace_run.cc``\ for full usage. The following list the key steps.

.. code:: cpp

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
