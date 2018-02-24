# **MACE** - *Mobile(Mi) Accelerated Compute Engine Library*
---

## Introduction
---
**Accelerating Neural Network with Heterogeneous Computing Devices in the phone.**

Supported Devices: **CPU(NEON)/GPU/DSP**.

## Architecture
---
- Use computational pattern of **DAG consisting of Ops**. 
- **Tensor** objects manage all data.
- **Workspace** manage all **Tensors**.

## GPU
---
Use **Image** object to optimize memory access and parallel computing based on OpenCL 2.0.

Design the corresponding **Image** format to optimize memory access for different Op algorithm.
Each pixel of **Image** object contains four elements(e.g. RGBA).
The Following is **Buffer** and **Image** format for all **Tensors**.
| Tensor| Buffer| Image| Explanation|
| --------- | :---------:|:--------:|:----:|
|Channel-Major Input/Output | NHWC | [W * (C+3)/4, N * H] | Default Input/Output format|
|Height-Major Input/Output | NHWC | [W * C, N * (H+3)/4] | Winograd Convolution format| 
|Width-Major Input/Output | NHWC | [(W+3)/4 * C, N * H] | Winograd Convolution format|
|Convolution Filter | HWOI | [H * W * RoundUp<4>(I), (O+3)/4]|Convolution filter format，There is no difference compared to [H*w*I, (O+3)/4]|
|Depthwise Convlution Filter | HWIM | [H * W * M, (I+3)/4]|Depthwise-Convolution filter format|
|1-D Argument | W | [(W+3)/4, 1] | 1D argument format, e.g. Bias|