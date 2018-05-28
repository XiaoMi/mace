// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class FoldedBatchNormOpTest : public OpsTestBase {};

namespace {
void CalculateScaleOffset(const std::vector<float> &gamma,
                          const std::vector<float> &beta,
                          const std::vector<float> &mean,
                          const std::vector<float> &var,
                          const float epsilon,
                          std::vector<float> *scale,
                          std::vector<float> *offset) {
  size_t size = gamma.size();
  for (size_t i = 0; i < size; ++i) {
    (*scale)[i] = gamma[i] / std::sqrt(var[i] + epsilon);
    (*offset)[i] = beta[i] - mean[i] * (*scale)[i];
  }
}

template <DeviceType D>
void Simple() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input", {1, 6, 2, 1},
                                  {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  std::vector<float> scale(1);
  std::vector<float> offset(1);
  CalculateScaleOffset({4.0f}, {2.0}, {10}, {11.67f}, 1e-3, &scale, &offset);
  net.AddInputFromArray<D, float>("Scale", {1}, scale);
  net.AddInputFromArray<D, float>("Offset", {1}, offset);

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<D, float>("Input", NHWC, "InputNCHW", NCHW);
    OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
        .Input("InputNCHW")
        .Input("Scale")
        .Input("Offset")
        .Output("OutputNCHW")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<D, float>("OutputNCHW", NCHW, "Output", NHWC);
  } else if (D == DeviceType::GPU) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, float>(&net, "Scale", "ScaleImage",
                            kernels::BufferType::ARGUMENT);
    BufferToImage<D, float>(&net, "Offset", "OffsetImage",
                            kernels::BufferType::ARGUMENT);

    OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
        .Input("InputImage")
        .Input("ScaleImage")
        .Input("OffsetImage")
        .Output("OutputImage")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  }

  // Check
  auto expected = CreateTensor<float>(
      {1, 6, 2, 1}, {-3.8543, -3.8543, -1.5125, -1.5125, 0.8291, 0.8291, 3.1708,
                     3.1708, 5.5125, 5.5125, 7.8543, 7.8543});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-4);
}
}  // namespace

TEST_F(FoldedBatchNormOpTest, SimpleCPU) { Simple<DeviceType::CPU>(); }

TEST_F(FoldedBatchNormOpTest, SimpleOPENCL) { Simple<DeviceType::GPU>(); }

TEST_F(FoldedBatchNormOpTest, SimpleRandomOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 10;
  index_t channels = 3 + rand_r(&seed) % 50;
  index_t height = 64;
  index_t width = 64;

  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input",
                                             {batch, height, width, channels});
  net.AddRandomInput<DeviceType::GPU, float>("Scale", {channels});
  net.AddRandomInput<DeviceType::GPU, float>("Offset", {channels});

  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);

  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("InputNCHW")
      .Input("Scale")
      .Input("Offset")
      .Output("OutputNCHW")
      .Finalize(net.NewOperatorDef());

  // run cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW, "Output",
                                                  NHWC);

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on opencl
  BufferToImage<DeviceType::GPU, float>(&net, "Input", "InputImage",
                                        kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<DeviceType::GPU, float>(&net, "Scale", "ScaleImage",
                                        kernels::BufferType::ARGUMENT);
  BufferToImage<DeviceType::GPU, float>(&net, "Offset", "OffsetImage",
                                        kernels::BufferType::ARGUMENT);

  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("InputImage")
      .Input("ScaleImage")
      .Input("OffsetImage")
      .Output("OutputImage")
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(DeviceType::GPU);
  net.Sync();

  ImageToBuffer<DeviceType::GPU, float>(&net, "OutputImage", "OPENCLOutput",
                                        kernels::BufferType::IN_OUT_CHANNEL);
  ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-5, 1e-4);
}

TEST_F(FoldedBatchNormOpTest, SimpleRandomHalfOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 10;
  index_t channels = 3 + rand_r(&seed) % 50;
  index_t height = 64;
  index_t width = 64;

  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input",
                                             {batch, height, width, channels});
  net.AddRandomInput<DeviceType::GPU, float>("Scale", {channels});
  net.AddRandomInput<DeviceType::GPU, float>("Offset", {channels});

  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);

  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("InputNCHW")
      .Input("Scale")
      .Input("Offset")
      .Output("OutputNCHW")
      .Finalize(net.NewOperatorDef());

  // run cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW, "Output",
                                                  NHWC);

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on opencl
  BufferToImage<DeviceType::GPU, half>(&net, "Input", "InputImage",
                                       kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<DeviceType::GPU, half>(&net, "Scale", "ScaleImage",
                                       kernels::BufferType::ARGUMENT);
  BufferToImage<DeviceType::GPU, half>(&net, "Offset", "OffsetImage",
                                       kernels::BufferType::ARGUMENT);

  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("InputImage")
      .Input("ScaleImage")
      .Input("OffsetImage")
      .Output("OutputImage")
      .AddIntArg("T", static_cast<int>(DataType::DT_HALF))
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(DeviceType::GPU);
  net.Sync();

  ImageToBuffer<DeviceType::GPU, float>(&net, "OutputImage", "OPENCLOutput",
                                        kernels::BufferType::IN_OUT_CHANNEL);
  ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-2, 1e-2);
}

TEST_F(FoldedBatchNormOpTest, ComplexRandomOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 10;
  index_t channels = 3 + rand_r(&seed) % 50;
  index_t height = 103;
  index_t width = 113;

  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input",
                                             {batch, height, width, channels});
  net.AddRandomInput<DeviceType::GPU, float>("Scale", {channels});
  net.AddRandomInput<DeviceType::GPU, float>("Offset", {channels});

  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);

  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("InputNCHW")
      .Input("Scale")
      .Input("Offset")
      .Output("OutputNCHW")
      .Finalize(net.NewOperatorDef());

  // run cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW, "Output",
                                                  NHWC);

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on opencl
  BufferToImage<DeviceType::GPU, float>(&net, "Input", "InputImage",
                                        kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<DeviceType::GPU, float>(&net, "Scale", "ScaleImage",
                                        kernels::BufferType::ARGUMENT);
  BufferToImage<DeviceType::GPU, float>(&net, "Offset", "OffsetImage",
                                        kernels::BufferType::ARGUMENT);

  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("InputImage")
      .Input("ScaleImage")
      .Input("OffsetImage")
      .Output("OutputImage")
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(DeviceType::GPU);

  ImageToBuffer<DeviceType::GPU, float>(&net, "OutputImage", "OPENCLOutput",
                                        kernels::BufferType::IN_OUT_CHANNEL);
  ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-5, 1e-4);
}

TEST_F(FoldedBatchNormOpTest, ComplexRandomHalfOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 10;
  index_t channels = 3 + rand_r(&seed) % 50;
  index_t height = 103;
  index_t width = 113;

  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input",
                                             {batch, height, width, channels});
  net.AddRandomInput<DeviceType::GPU, float>("Scale", {channels});
  net.AddRandomInput<DeviceType::GPU, float>("Offset", {channels});

  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);

  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("InputNCHW")
      .Input("Scale")
      .Input("Offset")
      .Output("OutputNCHW")
      .Finalize(net.NewOperatorDef());

  // run cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW, "Output",
                                                  NHWC);

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on opencl
  BufferToImage<DeviceType::GPU, half>(&net, "Input", "InputImage",
                                       kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<DeviceType::GPU, half>(&net, "Scale", "ScaleImage",
                                       kernels::BufferType::ARGUMENT);
  BufferToImage<DeviceType::GPU, half>(&net, "Offset", "OffsetImage",
                                       kernels::BufferType::ARGUMENT);

  OpDefBuilder("FoldedBatchNorm", "FoldedBatchNormTest")
      .Input("InputImage")
      .Input("ScaleImage")
      .Input("OffsetImage")
      .Output("OutputImage")
      .AddIntArg("T", static_cast<int>(DataType::DT_HALF))
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(DeviceType::GPU);

  ImageToBuffer<DeviceType::GPU, float>(&net, "OutputImage", "OPENCLOutput",
                                        kernels::BufferType::IN_OUT_CHANNEL);
  ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-2, 1e-2);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
