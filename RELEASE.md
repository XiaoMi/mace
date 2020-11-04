# v1.0.0 (2020-11-04)
## Support Quantization For MACE Micro
At the beginning of this year, we released MACE Micro to fully support ultra-low-power inference scenarios of mobile phones and IoT devices. In this version, we support quantization for MACE Micro and integrate CMSIS5 to support Cortex-M chips better.

## Support More Model Formats
We find more and more R&D engineers are using the PyTorch framework to train their models. In previous versions, MACE transformed the PyTorch model by using ONNX format as a bridge. In order to serve PyTorch developers better, we support direct transformation for PyTorch models in this version, which improves the performance of the model inference.
At the same time, we cooperated with [MEGVII](URL 'https://www.megvii.com/') company and support its [MegEngine](URL 'https://github.com/MegEngine/MegEngine') model format. If you trained your models by [MegEngine](URL 'https://github.com/MegEngine/MegEngine') framework, now you can use MACE to deploy the models on mobile phones or IoT devices.

## Support More Data Precision
Armv8.2 provides support for half-precision floating-point data processing instructions, in this version we support the fp16 precision computation by Armv8.2 fp16 instructions, which increases inference speed by roughly 40% for models such as mobilenet-v1 model.
The bfloat16 (Brain Floating Point) floating-point format is a computer number format occupying 16 bits in computer memory, we also support bfloat16 precision in this version, which increases inference speed by roughly 40% for models such as mobilenet-v1/2 model on some low-end chips.

## Others
In this version, we also add the following features:
1. Support more operators, such as `GroupNorm`, `ExtractImagePatches`, `Elu`, etc.
2. Optimize the performance of the framework and operators, such as the `Reduce` operator.
3. Support dynamic filter of conv2d/deconv2d.
4. Integrate MediaTek APU support on mt6873, mt6885, and mt6853.

## Acknowledgement
Thanks to the following guys who contribute code which makes MACE better.

@ZhangZhijing1, who contributed the bf16 code which was then committed by someone else.
@yungchienhsu, @Yi-Kai-Chen, @Eric-YK-Chen, @yzchen, @gasgallo, @lq, @huahang, @elswork, @LovelyBuggies, @freewym.

# v0.13.0 (2020-04-03)
## Support for MACE Micro
Compared with mobile devices such as mobile phones, micro-controllers are small, low-energy computing devices, which are often embedded in hardware that only needs basic computing, including household appliances and IoT devices. Billions of microcontrollers are produced every year. MACE adds micro-controller support to fully support ultra-low-power inference scenarios of mobile phones and IoT devices. MACE's micro-controller engine does not rely on any OS, heap memory allocation, C++ library or other third-party libraries except the math library. 

## Further Support For Quantization
MACE supports two kinds of quantization mechanisms: quantization-aware training and post-training quantization. In this version, we add a mixed-use of them. Furthermore, we support Armv8.2 dot product instruction for CPU quantization.

## Performance Optimization
MACE is continuously optimizing the performance. This time, we add ION buffer support for Qualcomm socs, which greatly improves the inference performance of models that need to switch between GPU and CPU. Moreover, we optimize the operators' performance such as `ResizeNearestNeighbor`, `Deconv`.

## Others
In this version, We support many new operators, `BatchMatMulV2` and `Select` operators for TensorFlow, `Deconv2d`, `Strided-Slice`, `Sigmoid` for Hexagon DSP and fix some bugs on validation and tuning.

## Acknowledgement
Thanks for the following guys who contribute code which makes MACE better.
gasgallo

# v0.12.0 (2019-11-17)
------
## Performance Optimization
We found that the lack of OP implementations on devices(GPU, Hexagon DSP, etc.) would lead to inefficient model execution, for  the memory synchronization between the device and the CPU consumed much time, so we added and enhanced some operators on the GPU( reshape, lpnorm, mvnorm, etc.) and Hexagon DSP (s2d, d2s, sub, etc.) to improve the efficiency of model execution.

## Further Support For Speech Recognition
In the last version, we supported the Kaldi framework. In Xiaomi we did a lot of work to support the speech recognition model,  including the support of flatten, unsample and other operators in onnx, as well as some bug fixes.

## CMake Support
MACE is continuously optimizing our compilation tools. This time, we support cmake compilation. Because of the use of ccache for acceleration, the compilation speed of cmake is much faster than the original bazel.
Related Docs: https://mace.readthedocs.io/en/latest/user_guide/basic_usage_cmake.html

## Others
In this version, We supported detection of perfomance regression by dana , and  “ gpu_queue_window” parameter is added to yml file,  to solve the UI jam problem caused by GPU task execution.
Related Docs: https://mace.readthedocs.io/en/latest/faq.html

## Acknowledgement
Thanks for the following guys who contribute code which make MACE better.

yungchienhsu, gasgallo, albu, yunikkk

# v0.10.0 (2019-01-03)
------
## Improvements
1. Support mixing usage of CPU and GPU.
2. Support ONNX format.
3. Support ARM Linux development board.
4. Support CPU quantization.
5. Update DSP library.
6. Add `Depthwise Deconvolution` of Caffe.
7. Add documents about debug and benchmark.
8. Bug fixed.

## Incompatible Changes
1. Remove all APIs in mace_runtime.h

## New APIs
1. Add GPUContext and GPUContextBuilder API.
2. Add MaceEngineConfig API.
3. Add MaceStatus API.
4. MaceTensor support data format.

## Acknowledgement
Thanks for the following guys who contribute code which make MACE better.

ByronHsu, conansherry, jackwish, herbakamil, tomaszkaliciak, oneTaken,
madhavajay, wayen820, idstein, newway1995.

# v0.9.0 (2018-07-20)
------
## Improvements
1. New work flow and documents.
2. Separate the model library from MACE library.
3. Reduce the size of static and dynamic library.
4. Support `ArgMax` Operations.
5. Support `Deconvolution` of Caffe.
6. Support NDK-17b.

## Incompatible Changes
1. Use file to store OpenCL tuned parameters and Add `SetOpenCLParameterPath` API.

## New APIs
1. Add a new `MaceEngine::Init` API with model data file.

## Bug Fixed
1. Not unmap the model data file when load model from files with CPU runtime.
2. 2D LWS tuning does not work.
3. Winograd convolution of GPU failed when open tuning.
4. Incorrect dynamic library of host.

## Acknowledgement
Thanks for the following guys who contribute code which make MACE better.

Zero King(@l2dy), James Bie(@JamesBie), Sun Aries(@SunAriesCN), Allen(@allen0125),
conansherry(@conansherry), 黎明灰烬(@jackwish)


# v0.8.0 (2018-05-31)
------
1. Change build and run tools
2. Handle runtime failure

# v0.7.0 (2018-05-18)
------
1. Change interface that report error type
2. Improve CPU performance
3. Merge CPU/GPU engine to on

# v0.6.3 (2018-05-21)
------
1. support `float` `data_type` when running in GPU


# v0.6.2 (2018-05-17)
------
* Return status instead of abort when allocate failed


# v0.6.0 (2018-04-04)
------
1. Change mace header interfaces, only including necessary methods.

