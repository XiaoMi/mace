# **MACE** - *Mobile(Mi) Accelerated Compute Engine Library*
---
小米自主研发的移动端神经网络加速引擎。

*可加入用户邮件组 mace-users@xiaomi.com*

## 简介
---
**利用端侧的异构计算设备加速神经网络模型。**

目前支持的端侧计算设备包括：**CPU(NEON)/GPU/DSP**.

## 架构
---
采用Op组成的有向无环图的计算模式，使用**Tensor**对象存储所有数据并进行统一管理。

## GPU
---
基于OpenCL 2.0实现，使用Image的存储格式优化内存访问和并行计算。
针对不同Op的算法，设计对应的Image存储格式来优化内存访问。

下面是针对不同**Tensor**类型对应的Buffer和Image的格式。
| Tensor类型 | Buffer格式 | Image格式 | 说明 |
| --------- | :---------:|:--------:|:----:|
|Channel-Major Input/Output | NHWC | [W * (C+3)/4, N * H] | 默认输入输出的格式|
|Height-Major Input/Output | NHWC | [W * C, N * (H+3)/4] | Winograd Convolution所用格式| 
|Width-Major Input/Output | NHWC | [(W+3)/4 * C, N * H] | Winograd Convolution所用格式|
|Convolution Filter | HWOI | [H * W * RoundUp<4>(I), (O+3)/4]|卷积核格式，尝试过[H*w*I, (O+3)/4]，两者性能没有区别|
|Depthwise Convlution Filter | HWIM | [H * W * M, (I+3)/4]|Depthwise卷积核格式|
|1-D Argument | W | [(W+3)/4, 1] | 一维参数格式，如Bias|