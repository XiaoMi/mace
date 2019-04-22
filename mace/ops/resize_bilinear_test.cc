// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include <vector>

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class ResizeBilinearTest : public OpsTestBase {};

TEST_F(ResizeBilinearTest, CPUResizeBilinearWOAlignCorners) {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;

  // Add input data
  std::vector<float> input(24);
  std::iota(begin(input), end(input), 0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 2, 4, 3}, input);
  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntsArg("size", {1, 2})
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>({1, 1, 2, 3}, {0, 1, 2, 6, 7, 8});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(ResizeBilinearTest, ResizeBilinearWAlignCorners) {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;

  // Add input data
  std::vector<float> input(24);
  std::iota(begin(input), end(input), 0);
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 2, 4, 3}, input);
  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntArg("align_corners", 1)
      .AddIntsArg("size", {1, 2})
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp();
  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>({1, 1, 2, 3}, {0, 1, 2, 9, 10, 11});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

namespace {
template <DeviceType D>
void TestRandomResizeBilinear() {
  testing::internal::LogToStderr();
  static unsigned int seed = time(NULL);
  for (int round = 0; round < 10; ++round) {
    int batch = 1 + rand_r(&seed) % 5;
    int channels = 1 + rand_r(&seed) % 100;
    int height = 1 + rand_r(&seed) % 100;
    int width = 1 + rand_r(&seed) % 100;
    int in_height = 1 + rand_r(&seed) % 100;
    int in_width = 1 + rand_r(&seed) % 100;
    int align_corners = rand_r(&seed) % 1;

    // Construct graph
    OpsTestNet net;
    // Add input data
    net.AddRandomInput<D, float>("Input",
                                 {batch, in_height, in_width, channels});
    net.TransformDataFormat<DeviceType::CPU, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

    OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
        .Input("InputNCHW")
        .Output("OutputNCHW")
        .AddIntArg("align_corners", align_corners)
        .AddIntsArg("size", {height, width})
        .Finalize(net.NewOperatorDef());
    // Run on CPU
    net.RunOp(DeviceType::CPU);
    net.TransformDataFormat<DeviceType::CPU, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

    auto expected = net.CreateTensor<float>();
    expected->Copy(*net.GetOutput("Output"));

    if (D == DeviceType::GPU) {
      OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
          .Input("Input")
          .Output("Output")
          .AddIntArg("align_corners", align_corners)
          .AddIntsArg("size", {height, width})
          .Finalize(net.NewOperatorDef());
      // Run
      net.RunOp(D);
    }
    // Check
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5,
                            1e-6);
  }
}

void TestQuantizedResizeBilinear() {
  testing::internal::LogToStderr();
  static unsigned int seed = time(NULL);
  for (int round = 0; round < 10; ++round) {
    int batch = 1 + rand_r(&seed) % 5;
    int channels = 1 + rand_r(&seed) % 100;
    int height = 1 + rand_r(&seed) % 100;
    int width = 1 + rand_r(&seed) % 100;
    int in_height = 1 + rand_r(&seed) % 100;
    int in_width = 1 + rand_r(&seed) % 100;
    int align_corners = rand_r(&seed) % 1;

    // Construct graph
    OpsTestNet net;
    // Add input data
    net.AddRandomInput<CPU, float>("Input",
                                   {batch, in_height, in_width, channels},
                                   false,
                                   false,
                                   true,
                                   -1.f,
                                   1.f);
    net.TransformDataFormat<DeviceType::CPU, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

    OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
        .Input("InputNCHW")
        .Output("OutputNCHW")
        .AddIntArg("align_corners", align_corners)
        .AddIntsArg("size", {height, width})
        .Finalize(net.NewOperatorDef());
    // Run on CPU
    net.RunOp(DeviceType::CPU);
    net.TransformDataFormat<DeviceType::CPU, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

    // run quantize
    OpDefBuilder("Quantize", "QuantizeInput")
        .Input("Input")
        .Output("QuantizedInput")
        .OutputType({DT_UINT8})
        .AddIntArg("T", DT_UINT8)
        .Finalize(net.NewOperatorDef());
    net.RunOp();

    OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
        .Input("QuantizedInput")
        .Output("QuantizedOutput")
        .AddIntArg("align_corners", align_corners)
        .AddIntsArg("size", {height, width})
        .OutputType({DT_UINT8})
        .AddIntArg("T", DT_UINT8)
        .Finalize(net.NewOperatorDef());
    net.RunOp();

    Tensor *eq_output = net.GetTensor("QuantizedInput");
    Tensor *q_output = net.GetTensor("QuantizedOutput");
    q_output->SetScale(eq_output->scale());
    q_output->SetZeroPoint(eq_output->zero_point());
    OpDefBuilder("Dequantize", "DeQuantizeTest")
        .Input("QuantizedOutput")
        .Output("DequantizedOutput")
        .OutputType({DT_FLOAT})
        .AddIntArg("T", DT_UINT8)
        .Finalize(net.NewOperatorDef());
    net.RunOp();

    // Check
    ExpectTensorSimilar<float>(*net.GetOutput("Output"),
                               *net.GetTensor("DequantizedOutput"), 0.01);
  }
}

}  // namespace

TEST_F(ResizeBilinearTest, OPENCLRandomResizeBilinear) {
  TestRandomResizeBilinear<DeviceType::GPU>();
}

TEST_F(ResizeBilinearTest, QuantizedResizeBilinear) {
  TestQuantizedResizeBilinear();
}

}  // namespace test
}  // namespace ops
}  // namespace mace
