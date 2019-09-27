基本使用方法
=============================

确保已安装所需环境 (refer to :doc:`../installation/env_requirement`).

清空工作目录
-------------------------------

构建前，清空工作目录

    .. code:: sh

        tools/clear_workspace.sh


编译引擎
-------------------------------

确保 CMake 已安装 

    .. code:: sh

        RUNTIME=GPU bash tools/cmake/cmake-build-armeabi-v7a.sh

编译安装位置为 ``build/cmake-build/armeabi-v7a``, 可以使用 libmace 静态库或者动态库。

除了 armeabi-v7，其他支持的 abi 包括: ``arm64-v8a``, ``arm-linux-gnueabihf``, ``aarch64-linux-gnu``, ``host``;
支持的目标设备 （RUNTIME) 包括: ``GPU``, ``HEXAGON``, ``HTA``, ``APU``.


转换模型
-------------------------------

撰写模型相关的 YAML 配置文件：

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



假设模型配置文件的路径是： ``../mace-models/mobilenet-v1/mobilenet-v1.yml``，执行：

    .. code:: yaml

        python tools/python/convert.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml

将会在 ``build/mobilenet_v1/model/`` 中产生 4 个文件

    .. code:: sh

        ├── mobilenet_v1.pb                (模型结构文件)
        ├── mobilenet_v1.data              (模型参数文件)
        ├── mobilenet_v1_index.html        (可视化文件，可在浏览器中打开)
        └── mobilenet_v1.pb_txt            (模型结构文本文件，可以 vim 进行查看)


除了 tensorflow，还支持训练平台: ``caffe``, ``onnx``;
除了 gpu， 亦可指定运行设备为 ``cpu``, ``dsp``。


模型测试与性能评测
-------------------------------

我们提供了模型测试与性能评测工具。

模型转换后，执行下面命令进行测试：

    .. code:: sh

        python tools/python/run_model.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml --validate

或下面命令进行性能评测：

    .. code:: sh

        python tools/python/run_model.py --config ../mace-models/mobilenet-v1/mobilenet-v1.yml --benchmark


这两个命令将会自动在目标设备上测试模型，如果在移动设备上测试，请确保已经连接上。
如果想查看详细日志，可以提高日志级别，例如指定选项 ``--vlog_level=2``


集成模型到应用
--------------------------------------

可以查看源码 \ ``mace/tools/mace_run.cc``\ 了解更多详情。下面简要介绍相关步骤：

.. code:: cpp

    // 添加头文件按
    #include "mace/public/mace.h"

    // 0. 指定目标设备
    DeviceType device_type = DeviceType::GPU;

    // 1. 运行配置
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

    // 2. 指定输入输出节点
    std::vector<std::string> input_names = {...};
    std::vector<std::string> output_names = {...};

    // 3. 创建引擎实例
    std::shared_ptr<mace::MaceEngine> engine;
    MaceStatus create_engine_status;
    
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

    // 4. 创建输入输出缓存
    std::map<std::string, mace::MaceTensor> inputs;
    std::map<std::string, mace::MaceTensor> outputs;
    for (size_t i = 0; i < input_count; ++i) {
      // Allocate input and output
      int64_t input_size =
          std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 1,
                          std::multiplies<int64_t>());
      auto buffer_in = std::shared_ptr<float>(new float[input_size],
                                              std::default_delete<float[]>());
      // 读取输入数据
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

    // 5. 执行模型
    MaceStatus status = engine.Run(inputs, &outputs);

更多信息可参考 :doc:`../../user_guide/advanced_usage_cmake`.
