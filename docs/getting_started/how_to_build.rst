How to build
============

模型格式支持
-------------

+--------------+------------------------------------------------------------------------------------------+
| 框架格式     | 支持情况                                                                                 |
+==============+==========================================================================================+
| TensorFlow   | 推荐使用1.4以上版本，否则可能达不到最佳性能 (考虑到后续Android NN，建议首选TensorFLow)   |
+--------------+------------------------------------------------------------------------------------------+
| Caffe        | 推荐使用1.0以上版本，低版本可能不支持，建议改用TensorFlow                                |
+--------------+------------------------------------------------------------------------------------------+
| MXNet        | 尚未支持                                                                                 |
+--------------+------------------------------------------------------------------------------------------+
| ONNX         | 尚未支持                                                                                 |
+--------------+------------------------------------------------------------------------------------------+

环境要求
---------

``mace``\ 提供了包含开发运行所需环境的docker镜像，镜像文件可以参考\ ``./docker/``\ 。启动命令：

.. code:: sh

    sudo docker pull cr.d.xiaomi.net/mace/mace-dev
    sudo docker run -it --rm --privileged -v /dev/bus/usb:/dev/bus/usb --net=host -v /local/path:/container/path cr.d.xiaomi.net/mace/mace-dev /bin/bash

如果用户希望配置开发机上的环境，可以参考如下环境要求：

+---------------------+-----------------+---------------------------------------------------------------------------------------------------+
| 软件                | 版本号          | 安装命令                                                                                          |
+=====================+=================+===================================================================================================+
| bazel               | >= 0.5.4        | -                                                                                                 |
+---------------------+-----------------+---------------------------------------------------------------------------------------------------+
| android-ndk         | r12c            | -                                                                                                 |
+---------------------+-----------------+---------------------------------------------------------------------------------------------------+
| adb                 | >= 1.0.32       | apt install -y android-tools-adb                                                                  |
+---------------------+-----------------+---------------------------------------------------------------------------------------------------+
| tensorflow          | 1.4.0           | pip install tensorflow==1.4.0                                                                     |
+---------------------+-----------------+---------------------------------------------------------------------------------------------------+
| scipy               | >= 1.0.0        | pip install scipy                                                                                 |
+---------------------+-----------------+---------------------------------------------------------------------------------------------------+
| jinja2              | >= 2.10         | pip install jinja2                                                                                |
+---------------------+-----------------+---------------------------------------------------------------------------------------------------+
| PyYaml              | >= 3.12         | pip install pyyaml                                                                                |
+---------------------+-----------------+---------------------------------------------------------------------------------------------------+
| docker(for caffe)   | >= 17.09.0-ce   | `install doc <https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository>`__   |
+---------------------+-----------------+---------------------------------------------------------------------------------------------------+

使用简介
--------

1. 获取最新tag的代码

**建议尽可能使用最新tag下的代码，以及不要直接使用master分支的最新代码。**

.. code:: sh

    git clone git@v9.git.n.xiaomi.com:deep-computing/mace.git

    # update
    git fetch --all --tags --prune

    # get latest tag version
    tag_name=`git describe --abbrev=0 --tags`

    # checkout to latest tag branch
    git checkout -b ${tag_name} tags/${tag_name}

2. 模型优化

-  Tensorflow

TensorFlow训练得到的模型进行一系列的转换，可以提升设备上的运行速度。TensorFlow提供了官方工具
`TensorFlow Graph Transform
Tool <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md>`__
来进行模型优化
(此工具Docker镜像中已经提供，也可以直接点击`下载 <http://cnbj1-inner-fds.api.xiaomi.net/mace/tool/transform_graph>`__\ 这个工具，用户亦可从官方源码编译\`)。以下分别是GPU模型和DSP模型的优化命令：

.. code:: sh

    # GPU模型:
    ./transform_graph \
        --in_graph=tf_model.pb \
        --out_graph=tf_model_opt.pb \
        --inputs='input' \
        --outputs='output' \
        --transforms='strip_unused_nodes(type=float, shape="1,64,64,3") 
            strip_unused_nodes(type=float, shape="1,64,64,3")
            remove_nodes(op=Identity, op=CheckNumerics)
            fold_constants(ignore_errors=true)
            fold_batch_norms
            fold_old_batch_norms
            strip_unused_nodes
            sort_by_execution_order'

    # DSP模型:
    ./transform_graph \
        --in_graph=tf_model.pb \
        --out_graph=tf_model_opt.pb \
        --inputs='input' \
        --outputs='output' \
        --transforms='strip_unused_nodes(type=float, shape="1,64,64,3") 
            strip_unused_nodes(type=float, shape="1,64,64,3")
            remove_nodes(op=Identity, op=CheckNumerics)
            fold_constants(ignore_errors=true)
            fold_batch_norms
            fold_old_batch_norms
            backport_concatv2
            quantize_weights(minimum_size=2)
            quantize_nodes
            strip_unused_nodes
            sort_by_execution_order'

-  Caffe

Caffe目前只支持最新版本，旧版本请使用Caffe的工具进行升级。

.. code:: bash

    # Upgrade prototxt
    $CAFFE_ROOT/build/tools/upgrade_net_proto_text MODEL.prototxt MODEL.new.prototxt

    # Upgrade caffemodel
    $CAFFE_ROOT/build/tools/upgrade_net_proto_binary MODEL.caffemodel MODEL.new.caffemodel

3. 生成模型静态库

模型静态库的生成需要使用目标机型，\ ***并且要求必须在目标SOC的机型上编译生成静态库。***

我们提供了\ ``mace_tools.py``\ 工具，可以将模型文件转换成静态库。\ ``tools/mace_tools.py``\ 使用步骤：



3.2 运行\ ``tools/mace_tools.py``\ 脚本

.. code:: sh

    # print help message
    # python tools/mace_tools.py --help
    # --config 配置文件的路径
    # --output_dir 编译结果的输出文件目录，默认为`./build`
    # --round 调用`examples/mace_run`运行模型的次数，默认为`1`
    # --tuning 对opencl的参数调参，该项通常只有开发人员用到，默认为`true`
    # --mode 运行模式，包含build/run/validate/merge/all/benchmark，默认为`all`

    # 仅编译模型和生成静态库
    python tools/mace_tools.py --config=models/config.yaml --mode=build

    # 测试模型的运行时间
    python tools/mace_tools.py --config=models/config.yaml --mode=run --round=1000

    # 对比编译好的模型在mace上与直接使用tensorflow或者caffe运行的结果，相似度使用`余弦距离表示`
    # 其中使用OpenCL设备，默认相似度大于等于`0.995`为通过；DSP设备下，相似度需要达到`0.930`。
    python tools/mace_tools.py --config=models/config.yaml --mode=run --round=1000

    # 将已编译好的多个模型合并成静态库
    # 比如编译了8个模型，决定使用其中2个模型，这时候可以不重新build，直接修改全局配置文件，合并生成静态库
    python tools/mace_tools.py --config=models/config.yaml --mode=merge

    # 运行以上所有项（可用于测试速度，建议 round=20）
    python tools/mace_tools.py --config=models/config.yaml --mode=all --round=1000

    # 模型Benchmark：查看每个Op的运行时间
    python tools/mace_tools.py --config=models/config.yaml --mode=benchmark

    # 查看模型运行时占用内存（如果有多个模型，可能需要注释掉一部分配置，只剩一个模型的配置）
    python tools/mace_tools.py --config=models/config.yaml --mode=run --round=10000 &
    adb shell dumpsys meminfo | grep mace_run
    sleep 10
    kill %1

4. 发布

通过前面的步骤，我们得到了包含业务模型的库文件。在业务代码中，我们只需要引入下面3组文件（\ ``./build/``\ 是默认的编译结果输出目录）：

头文件(包含mace.h和各个模型的头文件)： \*
``./build/${project_name}/${target_abi}/include/mace/public/*.h``

静态库（包含mace engine、opencl和模型相关库）： \*
``./build/${project_name}/${target_abi}/*.a``

动态库（仅编译的模型中包含dsp模式时用到）： \*
``./build/${project_name}/${target_abi}/libhexagon_controller.so``

模型数据文件（仅在EMBED\_MODEL\_DATA=0时产生）： \*
``./build/${project_name}/data/${MODEL_TAG}.data``

编译过程中间文件： \* ``./build/${project_name}/build/``

库文件tar包： \* ``./build/${project_name}/${project_name}.tar.gz``

5. 使用

具体使用流程可参考\ ``mace/examples/mace_run.cc``\ ，下面列出关键步骤。

.. code:: cpp

    // 引入头文件
    #include "mace/public/mace.h"
    #include "mace/public/{MODEL_TAG}.h"

    // 0. 设置内部存储
    const std::string file_path ="/path/to/store/internel/files";
    std::shared_ptr<KVStorageFactory> storage_factory(
        new FileStorageFactory(file_path));
    ConfigKVStorageFactory(storage_factory);

    //1. 从文件或代码中Load模型数据，也可通过自定义的方式来Load (例如可自己实现压缩加密等)
    // 如果使用的是数据嵌入的方式，将参数设为nullptr。
    unsigned char *model_data = mace::MACE_MODEL_TAG::LoadModelData(FLAGS_model_data_file.c_str());

    //2. 创建net对象
    NetDef net_def = mace::MACE_MODEL_TAG::CreateNet(model_data);

    //3. 声明设备类型(必须与build时指定的runtime一致）
    DeviceType device_type = DeviceType::OPENCL;

    //4. 定义输入输出名称数组
    std::vector<std::string> input_names = {...};
    std::vector<std::string> output_names = {...};

    //5. 创建输入输出对象
    std::map<std::string, mace::MaceTensor> inputs;
    std::map<std::string, mace::MaceTensor> outputs;
    for (size_t i = 0; i < input_count; ++i) {
      // Allocate input and output
      int64_t input_size =
          std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 1,
                          std::multiplies<int64_t>());
      auto buffer_in = std::shared_ptr<float>(new float[input_size],
                                              std::default_delete<float[]>());
      // load input
      ...

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

    //6. 创建MaceEngine对象
    mace::MaceEngine engine(&net_def, device_type, input_names, output_names);

    //7. 如果设备类型是OPENCL或HEXAGON，可以在此释放model_data
    if (device_type == DeviceType::OPENCL || device_type == DeviceType::HEXAGON) {
      mace::MACE_MODEL_TAG::UnloadModelData(model_data);
    }

    //8. 执行模型，得到结果
    engine.Run(inputs, &outputs);

