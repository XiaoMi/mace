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

class BiasAddOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void BiasAddSimple() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input", {1, 6, 2, 1},
                                  {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  net.AddInputFromArray<D, float>("Bias", {1}, {0.5f});

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    OpDefBuilder("BiasAdd", "BiasAddTest")
        .Input("InputNCHW")
        .Input("Bias")
        .AddIntArg("data_format", NCHW)
        .Output("OutputNCHW")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);
  } else if (D == DeviceType::GPU) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, float>(&net, "Bias", "BiasImage",
                            kernels::BufferType::ARGUMENT);

    OpDefBuilder("BiasAdd", "BiasAddTest")
        .Input("InputImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  // Check
  auto expected = CreateTensor<float>(
      {1, 6, 2, 1},
      {5.5, 5.5, 7.5, 7.5, 9.5, 9.5, 11.5, 11.5, 13.5, 13.5, 15.5, 15.5});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(BiasAddOpTest, BiasAddSimpleCPU) { BiasAddSimple<DeviceType::CPU>(); }

TEST_F(BiasAddOpTest, BiasAddSimpleOPENCL) { BiasAddSimple<DeviceType::GPU>(); }

TEST_F(BiasAddOpTest, SimpleRandomOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 10;
  index_t channels = 3 + rand_r(&seed) % 50;
  index_t height = 64 + rand_r(&seed) % 50;
  index_t width = 64 + rand_r(&seed) % 50;

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input",
                                             {batch, height, width, channels});
  net.AddRandomInput<DeviceType::GPU, float>("Bias", {channels}, true);

  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);

  // Construct graph
  OpDefBuilder("BiasAdd", "BiasAddTest")
      .Input("InputNCHW")
      .Input("Bias")
      .AddIntArg("data_format", NCHW)
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
  BufferToImage<DeviceType::GPU, float>(&net, "Bias", "BiasImage",
                                        kernels::BufferType::ARGUMENT);

  OpDefBuilder("BiasAdd", "BiasAddTest")
      .Input("InputImage")
      .Input("BiasImage")
      .Output("OutputImage")
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(DeviceType::GPU);
  net.Sync();

  ImageToBuffer<DeviceType::GPU, float>(&net, "OutputImage", "OPENCLOutput",
                                        kernels::BufferType::IN_OUT_CHANNEL);
  ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-5);
}

TEST_F(BiasAddOpTest, ComplexRandomOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 10;
  index_t channels = 3 + rand_r(&seed) % 50;
  index_t height = 103 + rand_r(&seed) % 100;
  index_t width = 113 + rand_r(&seed) % 100;

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input",
                                             {batch, height, width, channels});
  net.AddRandomInput<DeviceType::GPU, float>("Bias", {channels}, true);

  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);

  // Construct graph
  OpDefBuilder("BiasAdd", "BiasAddTest")
      .Input("InputNCHW")
      .Input("Bias")
      .AddIntArg("data_format", NCHW)
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
  BufferToImage<DeviceType::GPU, float>(&net, "Bias", "BiasImage",
                                        kernels::BufferType::ARGUMENT);

  OpDefBuilder("BiasAdd", "BiasAddTest")
      .Input("InputImage")
      .Input("BiasImage")
      .Output("OutputImage")
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(DeviceType::GPU);
  net.Sync();

  ImageToBuffer<DeviceType::GPU, float>(&net, "OutputImage", "OPENCLOutput",
                                        kernels::BufferType::IN_OUT_CHANNEL);
  ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-5);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
