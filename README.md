# MiAI Compute Engine
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![build status](http://v9.git.n.xiaomi.com/deep-computing/mace/badges/master/build.svg)](http://v9.git.n.xiaomi.com/deep-computing/mace/pipelines)

[Documentation](docs) |
[FAQ](docs/faq.md) |
[Release Notes](RELEASE.md) |
[MiAI Model Zoo](http://v9.git.n.xiaomi.com/deep-computing/mace-models) |
[Demo](mace/android) |
[中文](README_zh.md)

**MiAI Compute Engine** is a deep learning inference framework optimized for
mobile heterogeneous computing platforms. The design is focused on the following
targets:
* Performance
  * The runtime is highly optimized with NEON, OpenCL and Hexagon, and
    [Winograd algorithm](https://arxiv.org/abs/1509.09308) is introduced to
    speed up the convolution operations. Except for the inference speed, the
    initialization speed is also intensively optimized.
* Power consumption
  * Chip dependent power options like big.LITTLE scheduling, Adreno GPU hints are
    included as advanced API.
* Memory usage and library footprint
  * Graph level memory allocation optimization and buffer reuse is supported.
    The core library tries to keep minium external dependencies to keep the
    library footprint small.
* Model protection
  * Model protection is one the highest priority feature from the beginning of 
    the design. Various techniques are introduced like coverting models to C++
    code and literal obfuscations.
* Platform coverage
  * A good coverage of recent Qualcomm, MediaTek, Pinecone and other ARM based
    chips. CPU runtime is also compitable with most POSIX systems and
    archetectures with limited performance.

## Getting Started
* [Introduction](docs/getting_started/introduction.rst)
* [How to build](docs/getting_started/how_to_build.rst)
* [Create a model deployment file](docs/getting_started/create_a_model_deployment.rst)

## Performance
[MiAI Compute Engine Model Zoo](http://v9.git.n.xiaomi.com/deep-computing/mace-models) contains
several common neural networks models and built daily against a list of mobile
phones. The benchmark result can be found in the CI result page.

## Communication
* GitHub issues: bug reports, usage issues, feature requests
* Gitter:
* QQ群: 756046893

## Contributing
Any kind of contributions are welcome. For bug reports, feature requests,
please just open an issue without any hesitance. For code contributions, it's
strongly suggested to open an issue for discussion first. For more details,
please refer to [the contribution guide](docs/development/contributing.md).

## License
[Apache License 2.0](LICENSE).

## Acknowledgement
MiAI Compute Engine depends on several open source projects located in
[third_party](mace/third_party) directory. Particularly, we learned a lot from
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
their helps.
