# MACE - 移动人工智能计算引擎
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![build status](http://v9.git.n.xiaomi.com/deep-computing/mace/badges/master/build.svg)](http://v9.git.n.xiaomi.com/deep-computing/mace/pipelines)

[文档](docs) |
[FAQ](docs/faq.md) |
[发布记录](RELEASE.md) |
[MACE Model Zoo](https://github.com/XiaoMi/mace-models) |
[Demo](mace/examples/android) |
[English](README.md)

**Mobile AI Compute Engine (MACE)** 是一个专为移动端异构计算平台优化的神经网络计算框架。
主要从以下的角度做了专门的优化：
* 性能
  * 代码经过NEON指令，OpenCL以及Hexagon HVX专门优化，并且采用
  [Winograd算法](https://arxiv.org/abs/1509.09308)来进行卷积操作的加速。
  此外，还对启动速度进行了专门的优化。
* 功耗
  * 支持芯片的功耗管理，例如ARM的big.LITTLE调度，以及高通Adreno GPU功耗选项。
* 系统响应
  * 支持自动拆解长时间的OpenCL计算任务，来保证UI渲染任务能够做到较好的抢占调度，
  从而保证系统UI的相应和用户体验。
* 内存占用
  * 通过运用内存依赖分析技术，以及内存复用，减少内存的占用。另外，保持尽量少的外部
  依赖，保证代码尺寸精简。
* 模型加密与保护
  * 模型保护是重要设计目标之一。支持将模型转换成C++代码，以及关键常量字符混淆，增加逆向的难度。
* 硬件支持范围
  * 支持高通，联发科，以及松果等系列芯片的CPU，GPU与DSP(目前仅支持Hexagon)计算加速。
  同时支持在具有POSIX接口的系统的CPU上运行。

## 开始使用
* [简介](docs/getting_started/introduction.rst)
* [创建模型部署文件](docs/getting_started/create_a_model_deployment.rst)
* [如何构建](docs/getting_started/how_to_build.rst)

## 性能评测
[MACE Model Zoo](https://github.com/XiaoMi/mace-models)
包含若干常用模型，并且会对一组手机进行每日构建。最新的性能评测结果可以从项目的持续集成页面获取。

## 交流与反馈
* 欢迎通过Github Issues提交问题报告与建议
* 邮件列表: [mace-users@googlegroups.com](mailto:mace-users@googlegroups.com)
* QQ群: 756046893

## License
[Apache License 2.0](LICENSE).
