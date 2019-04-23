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

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class BatchNormOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void Simple() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("Input", {1, 6, 2, 1},
                                  {5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15});
  net.AddInputFromArray<D, float>("Scale", {1}, {4.0f}, true);
  net.AddInputFromArray<D, float>("Offset", {1}, {2.0}, true);
  net.AddInputFromArray<D, float>("Mean", {1}, {10}, true);
  net.AddInputFromArray<D, float>("Var", {1}, {11.67f}, true);

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<D, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
    OpDefBuilder("BatchNorm", "BatchNormTest")
        .Input("InputNCHW")
        .Input("Scale")
        .Input("Offset")
        .Input("Mean")
        .Input("Var")
        .AddFloatArg("epsilon", 1e-3)
        .Output("OutputNCHW")
        .Finalize(net.NewOperatorDef());
    // Run

    net.RunOp(D);
    net.TransformDataFormat<D, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  } else if (D == DeviceType::GPU) {
    OpDefBuilder("BatchNorm", "BatchNormTest")
        .Input("Input")
        .Input("Scale")
        .Input("Offset")
        .Input("Mean")
        .Input("Var")
        .AddFloatArg("epsilon", 1e-3)
        .Output("Output")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  }

  // Check
  auto expected = net.CreateTensor<float>(
      {1, 6, 2, 1}, {-3.8543, -3.8543, -1.5125, -1.5125, 0.8291, 0.8291, 3.1708,
                     3.1708, 5.5125, 5.5125, 7.8543, 7.8543});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-4);
}
}  // namespace

TEST_F(BatchNormOpTest, SimpleCPU) { Simple<DeviceType::CPU>(); }

TEST_F(BatchNormOpTest, SimpleOPENCL) { Simple<DeviceType::GPU>(); }

TEST_F(BatchNormOpTest, SimpleRandomOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 5;
  index_t channels = 3 + rand_r(&seed) % 25;
  index_t height = 64;
  index_t width = 64;

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input",
                                             {batch, height, width, channels});
  net.AddRandomInput<DeviceType::GPU, float>("Scale", {channels}, true, false);
  net.AddRandomInput<DeviceType::GPU, float>("Offset", {channels}, true, false);
  net.AddRandomInput<DeviceType::GPU, float>("Mean", {channels}, true);
  net.AddRandomInput<DeviceType::GPU, float>("Var", {channels}, true);

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  // Construct graph
  OpDefBuilder("BatchNorm", "BatchNormTest")
      .Input("InputNCHW")
      .Input("Scale")
      .Input("Offset")
      .Input("Mean")
      .Input("Var")
      .AddFloatArg("epsilon", 1e-3)
      .Output("OutputNCHW")
      .AddStringArg("activation", "LEAKYRELU")
      .AddFloatArg("leakyrelu_coefficient", 0.1)
      .Finalize(net.NewOperatorDef());

  // run cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("Output"));

  // Run on opencl
  OpDefBuilder("BatchNorm", "BatchNormTest")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .Input("Mean")
      .Input("Var")
      .AddFloatArg("epsilon", 1e-3)
      .Output("Output")
      .AddStringArg("activation", "LEAKYRELU")
      .AddFloatArg("leakyrelu_coefficient", 0.1)
      .Finalize(net.NewOperatorDef());

  // Tuning
  setenv("MACE_TUNING", "1", 1);
  net.RunOp(DeviceType::GPU);
  unsetenv("MACE_TUNING");

  // Run on opencl
  net.RunOp(DeviceType::GPU);
  net.Sync();
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"),
                          1e-5, 1e-4);
}

TEST_F(BatchNormOpTest, SimpleRandomHalfOPENCL) {
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
  net.AddRandomInput<DeviceType::GPU, float>("Scale", {channels}, true);
  net.AddRandomInput<DeviceType::GPU, float>("Offset", {channels}, true);
  net.AddRandomInput<DeviceType::GPU, float>("Mean", {channels}, true);
  net.AddRandomInput<DeviceType::GPU, float>("Var", {channels}, true);

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("BatchNorm", "BatchNormTest")
      .Input("InputNCHW")
      .Input("Scale")
      .Input("Offset")
      .Input("Mean")
      .Input("Var")
      .AddFloatArg("epsilon", 1e-1)
      .Output("OutputNCHW")
      .Finalize(net.NewOperatorDef());

  // run cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("Output"));

  // Run on opencl
  OpDefBuilder("BatchNorm", "BatchNormTest")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .Input("Mean")
      .Input("Var")
      .AddFloatArg("epsilon", 1e-1)
      .Output("Output")
      .AddIntArg("T", static_cast<int>(DataType::DT_HALF))
      .Finalize(net.NewOperatorDef());

  // Tuning
  setenv("MACE_TUNING", "1", 1);
  net.RunOp(DeviceType::GPU);
  unsetenv("MACE_TUNING");

  // Run on opencl
  net.RunOp(DeviceType::GPU);
  net.Sync();

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"),
                          1e-1, 1e-2);
}

TEST_F(BatchNormOpTest, ComplexRandomOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 5;
  index_t channels = 3 + rand_r(&seed) % 25;
  index_t height = 103;
  index_t width = 113;

  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input",
                                             {batch, height, width, channels});
  net.AddRandomInput<DeviceType::GPU, float>("Scale", {channels}, true);
  net.AddRandomInput<DeviceType::GPU, float>("Offset", {channels}, true);
  net.AddRandomInput<DeviceType::GPU, float>("Mean", {channels}, true);
  net.AddRandomInput<DeviceType::GPU, float>("Var", {channels}, true);

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("BatchNorm", "BatchNormTest")
      .Input("InputNCHW")
      .Input("Scale")
      .Input("Offset")
      .Input("Mean")
      .Input("Var")
      .AddFloatArg("epsilon", 1e-3)
      .Output("OutputNCHW")
      .Finalize(net.NewOperatorDef());

  // run cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("Output"));

  // Run on opencl
  OpDefBuilder("BatchNorm", "BatchNormTest")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .Input("Mean")
      .Input("Var")
      .AddFloatArg("epsilon", 1e-3)
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // tuning
  setenv("MACE_TUNING", "1", 1);
  net.RunOp(DeviceType::GPU);
  unsetenv("MACE_TUNING");

  // Run on opencl
  net.RunOp(DeviceType::GPU);
  net.Sync();

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"),
                          1e-5, 1e-4);
}

TEST_F(BatchNormOpTest, ComplexRandomHalfOPENCL) {
  // generate random input
  static unsigned int seed = time(NULL);
  index_t batch = 1 + rand_r(&seed) % 5;
  index_t channels = 3 + rand_r(&seed) % 25;
  index_t height = 103;
  index_t width = 113;

  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::GPU, float>("Input",
                                             {batch, height, width, channels});
  net.AddRandomInput<DeviceType::GPU, float>("Scale", {channels}, true);
  net.AddRandomInput<DeviceType::GPU, float>("Offset", {channels}, true);
  net.AddRandomInput<DeviceType::GPU, float>("Mean", {channels}, true);
  net.AddRandomInput<DeviceType::GPU, float>("Var", {channels}, true);

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("BatchNorm", "BatchNormTest")
      .Input("InputNCHW")
      .Input("Scale")
      .Input("Offset")
      .Input("Mean")
      .Input("Var")
      .AddFloatArg("epsilon", 1e-1)
      .Output("OutputNCHW")
      .Finalize(net.NewOperatorDef());

  // run cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  // Check
  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("Output"));

  // Run on opencl
  OpDefBuilder("BatchNorm", "BatchNormTest")
      .Input("Input")
      .Input("Scale")
      .Input("Offset")
      .Input("Mean")
      .Input("Var")
      .AddFloatArg("epsilon", 1e-1)
      .Output("Output")
      .AddIntArg("T", static_cast<int>(DataType::DT_HALF))
      .Finalize(net.NewOperatorDef());

  // tuning
  setenv("MACE_TUNING", "1", 1);
  net.RunOp(DeviceType::GPU);
  unsetenv("MACE_TUNING");

  // Run on opencl
  net.RunOp(DeviceType::GPU);
  net.Sync();

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"),
                          1e-1, 1e-2);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
