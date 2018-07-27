<div align="center">
<img src="docs/mace-logo.png" width="400" alt="MACE" />
</div>


[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Build Status](https://travis-ci.org/XiaoMi/mace.svg?branch=master)](https://travis-ci.org/XiaoMi/mace)
[![pipeline status](https://gitlab.com/llhe/mace/badges/master/pipeline.svg)](https://gitlab.com/llhe/mace/pipelines)
[![doc build status](https://readthedocs.org/projects/mace/badge/?version=latest)](https://readthedocs.org/projects/mace/badge/?version=latest)

[Documentation](https://mace.readthedocs.io) |
[FAQ](https://mace.readthedocs.io/en/latest/faq.html) |
[Release Notes](RELEASE.md) |
[Roadmap](ROADMAP.md) |
[MACE Model Zoo](https://github.com/XiaoMi/mace-models) |
[Demo](mace/examples/android) |
[Join Us](JOBS.md) |
[中文](README_zh.md)

**Mobile AI Compute Engine** (or **MACE** for short) is a deep learning inference framework optimized for
mobile heterogeneous computing platforms. The design focuses on the following
targets:
* Performance
  * Runtime is optimized with NEON, OpenCL and Hexagon, and
    [Winograd algorithm](https://arxiv.org/abs/1509.09308) is introduced to
    speed up convolution operations. The initialization is also optimized to be faster.
* Power consumption
  * Chip dependent power options like big.LITTLE scheduling, Adreno GPU hints are
    included as advanced APIs.
* Responsiveness
  * UI responsiveness guarantee is sometimes obligatory when running a model.
    Mechanism like automatically breaking OpenCL kernel into small units is
    introduced to allow better preemption for the UI rendering task.
* Memory usage and library footprint
  * Graph level memory allocation optimization and buffer reuse are supported.
    The core library tries to keep minimum external dependencies to keep the
    library footprint small.
* Model protection
  * Model protection has been the highest priority since the beginning of 
    the design. Various techniques are introduced like converting models to C++
    code and literal obfuscations.
* Platform coverage
  * Good coverage of recent Qualcomm, MediaTek, Pinecone and other ARM based
    chips. CPU runtime is also compatible with most POSIX systems and
    architectures with limited performance.

## Getting Started
* [Introduction](https://mace.readthedocs.io/en/latest/introduction.html)
* [Installation](https://mace.readthedocs.io/en/latest/installation/env_requirement.html)
* [Basic Usage](https://mace.readthedocs.io/en/latest/user_guide/basic_usage.html)
* [Advanced Usage](https://mace.readthedocs.io/en/latest/user_guide/advanced_usage.html)

## Performance
[MACE Model Zoo](https://github.com/XiaoMi/mace-models) contains
several common neural networks and models which will be built daily against a list of mobile
phones. The benchmark results can be found in [the CI result page](https://gitlab.com/llhe/mace-models/pipelines)
(choose the latest passed pipeline, click *release* step and you will see the benchmark results).
To get the comparison results with other frameworks, you can take a look at
[MobileAIBench](https://github.com/XiaoMi/mobile-ai-bench) project.

## Communication
* GitHub issues: bug reports, usage issues, feature requests
* Slack: [mace-users.slack.com](https://join.slack.com/t/mace-users/shared_invite/enQtMzkzNjM3MzMxODYwLTAyZTAzMzQyNjc0ZGI5YjU3MjI1N2Q2OWI1ODgwZjAwOWVlNzFlMjFmMTgwYzhjNzU4MDMwZWQ1MjhiM2Y4OTE)
* QQ群: 756046893

## Contributing
Any kind of contribution is welcome. For bug reports, feature requests,
please just open an issue without any hesitation. For code contributions, it's
strongly suggested to open an issue for discussion first. For more details,
please refer to [the contribution guide](https://mace.readthedocs.io/en/latest/development/contributing.html).

## License
[Apache License 2.0](LICENSE).

## Acknowledgement
MACE depends on several open source projects located in the
[third_party](third_party) directory. Particularly, we learned a lot from
the following projects during the development:
* [Qualcomm Hexagon NN Offload Framework](https://source.codeaurora.org/quic/hexagon_nn/nnlib): the Hexagon DSP runtime
  depends on this library.
* [TensorFlow](https://github.com/tensorflow/tensorflow),
  [Caffe](https://github.com/BVLC/caffe),
  [SNPE](https://developer.qualcomm.com/software/snapdragon-neural-processing-engine-ai),
  [ARM ComputeLibrary](https://github.com/ARM-software/ComputeLibrary),
  [ncnn](https://github.com/Tencent/ncnn) and many others: we learned many best
  practices from these projects.

Finally, we also thank the Qualcomm, Pinecone and MediaTek engineering teams for
their help.

## Join Us
[We are hiring](JOBS.md).
