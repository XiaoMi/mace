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

class LpNormOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestLpNorm(const std::vector<index_t> &input_shape,
                const std::vector<T> &input,
                const int p,
                const int axis,
                const std::vector<T> &output) {
  OpsTestNet net;
  net.AddInputFromArray<D, T>(MakeString("Input"), input_shape, input);

  if (D == DeviceType::GPU) {
    net.TransformDataFormat<GPU, float>(
        "Input", DataFormat::NCHW, "InputNHWC", DataFormat::NHWC);
  }

  OpDefBuilder("LpNorm", "LpNormTest")
      .Input(D == DeviceType::CPU ? "Input" : "InputNHWC")
      .AddIntArg("p", p)
      .AddIntArg("axis", axis)
      .Output(D == DeviceType::CPU ? "Output" : "OutputNHWC")
      .Finalize(net.NewOperatorDef());

  net.RunOp(D);

  if (D == DeviceType::GPU) {
    net.TransformDataFormat<GPU, float>(
        "OutputNHWC", DataFormat::NHWC, "Output", DataFormat::NCHW);
  }

  net.AddInputFromArray<D, T>("ExpectedOutput", input_shape, output);
  ExpectTensorNear<T>(*net.GetOutput("ExpectedOutput"),
                      *net.GetOutput("Output"));
}
}  // namespace

TEST_F(LpNormOpTest, SimpleTestFabs) {
  TestLpNorm<DeviceType::CPU, float>(
    {1, 8, 1, 2},  // NCHW
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    1, 1,
    {0.00735294, 0.0147059, 0.0220588, 0.0294118,
     0.0367647, 0.0441176, 0.0514706, 0.0588235,
     0.0661765, 0.0735294, 0.0808824, 0.0882353,
     0.0955882, 0.102941, 0.110294, 0.117647});
}

TEST_F(LpNormOpTest, SimpleTestSquare) {
  TestLpNorm<DeviceType::CPU, float>(
    {1, 8, 1, 2},  // NCHW
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    2, 1,
    {0.0258544, 0.0517088, 0.0775632, 0.103418,
     0.129272, 0.155126, 0.180981, 0.206835,
     0.232689, 0.258544, 0.284398, 0.310253,
     0.336107, 0.361961, 0.387816, 0.41367});
}

TEST_F(LpNormOpTest, SimpleTestPSquare2) {
TestLpNorm<DeviceType::CPU, float>(
    {1, 8, 1, 2},  // NCHW
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    2, 2,
    {0.447214, 0.894427, 0.600000, 0.800000,
     0.640184, 0.768221, 0.658505, 0.752577,
     0.668965, 0.743294, 0.675725, 0.737154,
     0.680451, 0.732793, 0.683941, 0.729537});
}

TEST_F(LpNormOpTest, SimpleTestFabsOpenCL) {
  TestLpNorm<DeviceType::GPU, float>(
    {1, 8, 1, 2},  // NCHW
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    1, 1,
    {0.00735294, 0.0147059, 0.0220588, 0.0294118,
     0.0367647, 0.0441176, 0.0514706, 0.0588235,
     0.0661765, 0.0735294, 0.0808824, 0.0882353,
     0.0955882, 0.102941, 0.110294, 0.117647});
}

TEST_F(LpNormOpTest, SimpleTestSquareOpenCL) {
  TestLpNorm<DeviceType::GPU, float>(
    {1, 8, 1, 2},  // NCHW
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    2, 1,
    {0.0258544, 0.0517088, 0.0775632, 0.103418,
     0.129272, 0.155126, 0.180981, 0.206835,
     0.232689, 0.258544, 0.284398, 0.310253,
     0.336107, 0.361961, 0.387816, 0.41367});
}

TEST_F(LpNormOpTest, SimpleTestSquareOpenCL2) {
  TestLpNorm<DeviceType::GPU, float>(
    {1, 8, 1, 2},  // NCHW
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    2, 2,
    {0.447214, 0.894427, 0.600000, 0.800000,
     0.640184, 0.768221, 0.658505, 0.752577,
     0.668965, 0.743294, 0.675725, 0.737154,
     0.680451, 0.732793, 0.683941, 0.729537});
}


namespace {
template <DeviceType D, typename T>
void TestLpNormRandom(const std::vector<index_t> &input_shape,
                      const int p,
                      const int axis) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", input_shape);

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("LpNorm", "LpNormTest")
      .Input("InputNCHW")
      .Output("OutputNCHW")
      .AddIntArg("p", p)
      .AddIntArg("axis", axis)
      .Finalize(net.NewOperatorDef());

  // run on cpu
  net.RunOp();

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("Output"));

  OpDefBuilder("LpNorm", "LpNormTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("p", p)
      .AddIntArg("axis", axis)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());
  net.RunOp(D);

  if (DataTypeToEnum<T>::value == DT_HALF) {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-2, 1e-3);
  } else {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
  }
}

}  // namespace

TEST_F(LpNormOpTest, SimpleTestSquareHalfOpenCL) {
  TestLpNormRandom<DeviceType::GPU, half>({1, 8, 1, 2}, 2, 1);
}

TEST_F(LpNormOpTest, SimpleTestSquareHalfOpenCL2) {
  TestLpNormRandom<DeviceType::GPU, half>({1, 8, 1, 2}, 2, 2);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
