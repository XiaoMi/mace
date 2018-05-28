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

class PadTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void Simple() {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRepeatedInput<D, float>("Input", {1, 2, 3, 1}, 2);
  if (D == DeviceType::GPU) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);
    OpDefBuilder("Pad", "PadTest")
        .Input("InputImage")
        .Output("OutputImage")
        .AddIntsArg("paddings", {0, 0, 1, 2, 1, 2, 0, 0})
        .AddFloatArg("constant_value", 1.0)
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);

    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "TInput",
                                                    NCHW);
    OpDefBuilder("Pad", "PadTest")
        .Input("TInput")
        .Output("TOutput")
        .AddIntsArg("paddings", {0, 0, 0, 0, 1, 2, 1, 2})
        .AddFloatArg("constant_value", 1.0)
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp();

    net.TransformDataFormat<DeviceType::CPU, float>("TOutput", NCHW, "Output",
                                                    NHWC);
  }

  auto output = net.GetTensor("Output");

  auto expected = CreateTensor<float>(
      {1, 5, 6, 1}, {
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2,   2,   2,
                        1.0, 1.0, 1.0, 2,   2,   2,   1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    });
  ExpectTensorNear<float>(*expected, *output, 1e-5);
}
}  // namespace

TEST_F(PadTest, SimpleCPU) { Simple<DeviceType::CPU>(); }

TEST_F(PadTest, SimpleGPU) { Simple<DeviceType::GPU>(); }

TEST_F(PadTest, ComplexCPU) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRepeatedInput<DeviceType::CPU, float>("Input", {1, 1, 1, 2}, 2);
  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "TInput",
                                                  NCHW);
  OpDefBuilder("Pad", "PadTest")
      .Input("TInput")
      .Output("TOutput")
      .AddIntsArg("paddings", {0, 0, 1, 1, 1, 1, 1, 1})
      .AddFloatArg("constant_value", 1.0)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>("TOutput", NCHW, "Output",
                                                  NHWC);

  auto output = net.GetTensor("Output");

  auto expected = CreateTensor<float>(
      {1, 3, 3, 4},
      {
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      });
  ExpectTensorNear<float>(*expected, *output, 1e-5);
}

namespace {
template <typename T>
void Complex(const std::vector<index_t> &input_shape,
             const std::vector<int> &cpu_paddings,
             const std::vector<int> &gpu_paddings) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input", input_shape);

  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "TInput",
                                                  NCHW);
  OpDefBuilder("Pad", "PadTest")
      .Input("TInput")
      .Output("TOutput")
      .AddIntsArg("paddings", cpu_paddings)
      .AddFloatArg("constant_value", 1.0)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>("TOutput", NCHW, "Output",
                                                  NHWC);

  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  BufferToImage<DeviceType::GPU, T>(&net, "Input", "InputImage",
                                    kernels::BufferType::IN_OUT_CHANNEL);
  OpDefBuilder("Pad", "PadTest")
      .Input("InputImage")
      .Output("OutputImage")
      .AddIntsArg("paddings", gpu_paddings)
      .AddFloatArg("constant_value", 1.0)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::GPU);

  ImageToBuffer<DeviceType::GPU, float>(&net, "OutputImage", "OpenCLOutput",
                                        kernels::BufferType::IN_OUT_CHANNEL);

  auto output = net.GetTensor("OpenCLOutput");

  if (DataTypeToEnum<T>::value == DT_HALF) {
    ExpectTensorNear<float>(expected, *output, 1e-2, 1e-2);
  } else {
    ExpectTensorNear<float>(expected, *output, 1e-5);
  }
}
}  // namespace

TEST_F(PadTest, ComplexFloat) {
  Complex<float>({1, 32, 32, 4}, {0, 0, 0, 0, 2, 2, 1, 1},
                 {0, 0, 2, 2, 1, 1, 0, 0});
  Complex<float>({1, 31, 37, 16}, {0, 0, 0, 0, 2, 0, 1, 0},
                 {0, 0, 2, 0, 1, 0, 0, 0});
  Complex<float>({1, 128, 128, 32}, {0, 0, 0, 0, 0, 1, 0, 2},
                 {0, 0, 0, 1, 0, 2, 0, 0});
}

TEST_F(PadTest, ComplexHalf) {
  Complex<half>({1, 32, 32, 4}, {0, 0, 0, 0, 2, 2, 1, 1},
                {0, 0, 2, 2, 1, 1, 0, 0});
  Complex<half>({1, 31, 37, 16}, {0, 0, 0, 0, 2, 0, 1, 0},
                {0, 0, 2, 0, 1, 0, 0, 0});
  Complex<half>({1, 128, 128, 32}, {0, 0, 0, 0, 0, 1, 0, 2},
                {0, 0, 0, 1, 0, 2, 0, 0});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
