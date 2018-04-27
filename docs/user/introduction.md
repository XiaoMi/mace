Introduction
============

**MACE** - *Mobile(Mi) Accelerated Compute Engine Library* 是小米自主研发的移动端神经网络加速引擎。

## 特点
1. 速度快
  * 专门为小米手机SoC优化(高通，MTK，澎湃)，支持GPU(DSP基于nnlib)加速，在主流高通平台速度优于高通SNPE框架 (注: 速度跟模型结构有关，depthwise conv2d，1x1卷积MACE与SNPE持平，3x3卷积，通用卷积等优于SNPE。另外目前高通平台SNPE明显优于TensorFlow Lite, Caffe/Caffe2，ARM Compute Library，腾讯ncnn，百度MDL等开源框架)。
  * 支持不同SoC的自动调优
  * 模型数据通过mmap方式加载，启动速度快
2. 内存占用少
  * MACE支持基于计算图依赖的内存优化技术，通过内存复用，能减少运行时内存占用，特别是对于依赖较简单的模型，内存优化效果明显
3. 体积小
  * MACE本身无外部依赖，核心代码小于1MB (模型除外)
4. 内置模型加密功能
  * MACE支持模型混淆加密功能，模型直接编译成可执行代码而非数据文件，同时加强了敏感代码的混淆，增加了反向的难度
5. 部署便捷
  * 用户接口简单，只需要一个头文件，MACE采用源码/静态库形式链接到用户程序，不会引入额外的动态库和模型数据文件(DSP版本需要一个额外的动态库)

## 模型格式支持
| 框架格式       | 支持情况 |
| ---------- |:-------:|
| TensorFlow | 推荐使用1.4以上版本，否则可能达不到最佳性能 (考虑到后续Android NN，建议首选TensorFLow) |
| Caffe | 推荐使用1.0以上版本，低版本可能不支持，建议改用TensorFlow |
| MXNet | 尚未支持 |
| ONNX | 尚未支持 |


## 环境要求

`mace`提供了包含开发运行所需环境的docker镜像，镜像文件可以参考`./docker/`。启动命令：
```sh
sudo docker pull cr.d.xiaomi.net/mace/mace-dev
sudo docker run -it --rm --privileged -v /dev/bus/usb:/dev/bus/usb --net=host -v /local/path:/container/path cr.d.xiaomi.net/mace/mace-dev /bin/bash
```

如果用户希望配置开发机上的环境，可以参考如下环境要求：

| 软件     | 版本号         | 安装命令 |
| -------- |:--------------:|:---------------------:|
| bazel | >= 0.5.4 | - |
| android-ndk | r12c | - |
| adb | >= 1.0.32 | apt install -y android-tools-adb |
| tensorflow | 1.4.0 | pip install tensorflow==1.4.0 |
| scipy | >= 1.0.0 | pip install scipy |
| jinja2 | >= 2.10 | pip install jinja2 |
| PyYaml | >= 3.12 | pip install pyyaml |
| docker(for caffe) | >= 17.09.0-ce | [install doc](https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository) |

## 文件组织

```
|-- tools --> mace编译运行相关的工具脚本
|   |-- mace_tools.py
|   |-- ...
|
|-- mace
|   |-- benchmark
|   |
|   |-- codegen --> 模型、opencl二进制文件和tuning数据生成的C++代码
|   |   |-- models
|   |   |-- opencl
|   |   |-- opencl_bin
|   |   |-- tuning
|   |
|   |-- core
|   |
|   |-- examples
|   |   |-- mace_run.cc --> 运行mace模型的样例
|   |   |-- ...
|   |
|   |-- kernels
|   |
|   |-- ops
|   |
|   |-- public --> mace的接口
|
|-- docker --> mace开发环境的Dockerfile
```

## 使用简介

1\. 获取最新tag的代码

**建议尽可能使用最新tag下的代码，以及不要直接使用master分支的最新代码。**

```sh
git clone git@v9.git.n.xiaomi.com:deep-computing/mace.git

# update
git fetch --all --tags --prune

# get latest tag version
tag_name=`git describe --abbrev=0 --tags`

# checkout to latest tag branch
git checkout -b ${tag_name} tags/${tag_name}
```

2\. 模型优化

- Tensorflow

TensorFlow训练得到的模型进行一系列的转换，可以提升设备上的运行速度。TensorFlow提供了官方工具 
[TensorFlow Graph Transform Tool](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md) 
来进行模型优化 (在官方工具的基础上，我们做了一部分定制化，此工具Docker镜像中已经提供，也可以直接点击[下载](http://cnbj1-inner-fds.api.xiaomi.net/mace/tool/transform_graph)这个工具，`暂时不支持用户自己从官方源码编译`)。以下分别是GPU模型和DSP模型的优化命令：

```sh
# GPU模型:
./transform_graph \
    --in_graph=tf_model.pb \
    --out_graph=tf_model_opt.pb \
    --inputs='input' \
    --outputs='output' \
    --transforms='strip_unused_nodes(type=float, shape="1,64,64,3") 
        strip_unused_nodes(type=float, shape="1,64,64,3")
        remove_nodes(op=Identity, op=CheckNumerics)
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
```

- Caffe

Caffe目前只支持最新版本，旧版本请使用Caffe的工具进行升级。
```bash
# Upgrade prototxt
$CAFFE_ROOT/build/tools/upgrade_net_proto_text MODEL.prototxt MODEL.new.prototxt

# Upgrade caffemodel
$CAFFE_ROOT/build/tools/upgrade_net_proto_binary MODEL.caffemodel MODEL.new.caffemodel

```

3\. 生成模型静态库

模型静态库的生成需要使用目标机型，***并且要求必须在目标SOC的机型上编译生成静态库。***


我们提供了`mace_tools.py`工具，可以将模型文件转换成静态库。`tools/mace_tools.py`使用步骤：


3\.1 配置文件

配置文件使用yml文件格式，配置项如下：
```yaml
# 配置文件名会被用作生成库的名称：libmace-${filename}.a
target_abis: [armeabi-v7a, arm64-v8a]
# 具体机型的soc编号，可以使用`adb shell getprop | grep ro.board.platform | cut -d [ -f3 | cut -d ] -f1`获取
target_socs: [msm8998]
embed_model_data: 1
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
```

具体配置项含义如下表：

| 配置项     |      含义      |
| ---------- |:--------------:|
| target_abis | 运行的ABI，可选包括安卓设备的armeabi-v7a，arm64-v8a等，以及开发人员的电脑终端（电脑终端使用‘host’表示）。可以同时指定多个ABI |
| embed_model_data | 是否将模型里的数据嵌入到代码中，默认为1 |
| platform | 模型对应的框架名称 [tensorflow | caffe] |
| model_file_path | 模型的路径，可以是一个http或https的下载链接 |
| weight_file_path | 权重文件的路径，可以是一个http或https的下载链接(caffe model)|
| model_sha256_checksum | The SHA256 checksum of the model file |
| weight_sha256_checksum | The SHA256 checksum of the weight file(caffe model) |
| input_nodes | 优化后的模型或其他框架模型的输入节点, 支持多个节点|
| output_nodes | 优化后的模型或其他框架模型的输出节点, 支持多个节点|
| input_shapes | 格式: NHWC. 模型的输入shape, 支持多个shape|
| output_shapes | 格式: NHWC. 模型的输出shape, 支持多个shape|
| runtime | 运行的设备，可选包含cpu、gpu和dsp |
| limit_opencl_kernel_time | 限制opencl的kernel每个work group运行时间在1ms以内，可能影响性能，默认关闭 |
| dsp_mode | 配置dsp的不同计算方式，以获得不同的精度和性能，一般使用默认值0即可 |
| obfuscate | 是否混淆模型内部各个操作的名称 |
| fast_conv| 使用最快的卷积算法，**可能会导致内存增多**|
| input_files| (可选). 指定模型输入文件，用于结果验证，必须与input_nodes对应。如未指定，则使用[-1,1]的随机值|

3\.2 运行`tools/mace_tools.py`脚本
```sh
# print help message
# python tools/mace_tools.py --help
# --config 配置文件的路径
# --output_dir 编译结果的输出文件目录，默认为`./build`
# --round 调用`examples/mace_run`运行模型的次数，默认为`1`
# --tuning 对opencl的参数调参，该项通常只有开发人员用到，默认为`true`
# --mode 运行模式，包含build/run/validate/merge/all/benchmark，默认为`all`

# 仅编译模型和生成静态库
python tools/mace_tools.py --config=models/config.yml --mode=build

# 测试模型的运行时间
python tools/mace_tools.py --config=models/config.yml --mode=run --round=1000

# 对比编译好的模型在mace上与直接使用tensorflow或者caffe运行的结果，相似度使用`余弦距离表示`
# 其中使用OpenCL设备，默认相似度大于等于`0.995`为通过；DSP设备下，相似度需要达到`0.930`。
python tools/mace_tools.py --config=models/config.yml --mode=run --round=1000

# 将已编译好的多个模型合并成静态库
# 比如编译了8个模型，决定使用其中2个模型，这时候可以不重新build，直接修改全局配置文件，合并生成静态库
python tools/mace_tools.py --config=models/config.yml --mode=merge

# 运行以上所有项（可用于测试速度，建议 round=20）
python tools/mace_tools.py --config=models/config.yml --mode=all --round=1000

# 模型Benchmark：查看每个Op的运行时间
python tools/mace_tools.py --config=models/config.yml --mode=benchmark

# 查看模型运行时占用内存（如果有多个模型，可能需要注释掉一部分配置，只剩一个模型的配置）
python tools/mace_tools.py --config=models/config.yml --mode=run --round=10000 &
adb shell dumpsys meminfo | grep mace_run
sleep 10
kill %1

```

4\. 发布

通过前面的步骤，我们得到了包含业务模型的库文件。在业务代码中，我们只需要引入下面3组文件（`./build/`是默认的编译结果输出目录）：

头文件(包含mace.h和各个模型的头文件)：
  * `./build/${project_name}/${target_abi}/include/mace/public/*.h`

静态库（包含mace engine、opencl和模型相关库）：
  * `./build/${project_name}/${target_abi}/*.a`

动态库（仅编译的模型中包含dsp模式时用到）：
  * `./build/${project_name}/${target_abi}/libhexagon_controller.so`

模型数据文件（仅在EMBED_MODEL_DATA=0时产生）：
  * `./build/${project_name}/data/${MODEL_TAG}.data`

编译过程中间文件：
  * `./build/${project_name}/build/`

库文件tar包：
  * `./build/${project_name}/${project_name}.tar.gz`
  
5\. 使用

具体使用流程可参考`mace/examples/mace_run.cc`，下面列出关键步骤。

```c++
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

//3. 声明设备类型
DeviceType device_type = DeviceType::GPU;

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

//7. 如果设备类型是GPU或者HEXAGON，可以在此释放model_data
if (device_type == DeviceType::GPU || device_type == DeviceType::HEXAGON) {
  mace::MACE_MODEL_TAG::UnloadModelData(model_data);
}

//8. 执行模型，得到结果
engine.Run(inputs, &outputs);

```

## 功能列表
算子持续完善中，有新功能需求请联系我们。

| 操作          | Android NN    | 状态      | 备注          |
| ------------- |:-------------:|:---------:|:-------------:|
| ADD | Y | Y | |
| AVERAGE_POOL_2D | Y | Y | |
| BATCH_NORM | | Y | 支持与激活层合并 |
| BIAS_ADD | | Y | |
| CHANNEL_SHUFFLE | | | |
| CONCATENATION | Y | Y | |
| CONV_2D | Y | Y | 支持stride，dilations，支持与batch norm和激活层合并 |
| DEPTHWISE_CONV_2D | Y | Y | 目前支持multiplier = 1以及与batch norm和激活层合并 |
| DEPTH_TO_SPACE | Y | | |
| DEQUANTIZE | Y | | |
| EMBEDDING_LOOKUP | Y | | |
| FLOOR | Y | | |
| FULLY_CONNECTED | Y | Y | |
| GROUP_CONV_2D | | | |
| HASHTABLE_LOOKUP | Y | | |
| L2_NORMALIZATION | Y | | |
| L2_POOL_2D | Y | | |
| LOCAL_RESPONSE_NORMALIZATION | Y | | |
| LOGISTIC | Y | Y | |
| LSH_PROJECTION | Y | | |
| LSTM | Y | | |
| MATMUL | | |  |
| MAX_POOL_2D | Y | Y | |
| MUL | Y | | |
| PSROI_ALIGN | | | |
| PRELU | | Y |  |
| RELU | Y | Y | |
| RELU1 | Y | Y | |
| RELU6 | Y | Y | |
| RELUX |  | Y | |
| RESHAPE | Y | | |
| RESIZE_BILINEAR | Y | Y | |
| RNN | Y | | |
| RPN_PROPOSAL_LAYER | | | |
| SOFTMAX | Y | Y | |
| SPACE_TO_DEPTH | Y | | |
| SVDF | Y | | |
| TANH | Y | Y | |


## 性能对比
待整理
