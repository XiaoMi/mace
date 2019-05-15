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

class CumsumOpTest : public OpsTestBase {};

namespace {
template <typename T>
void SimpleTestWithDataFormat(const std::vector<index_t> &shape,
                              const std::vector<float> &input,
                              const int axis,
                              const int exclusive,
                              const int reverse,
                              const std::vector<float> &output) {
  // Construct graph
  OpsTestNet net;

  net.AddInputFromArray<CPU, T>("Input", shape, input);
  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);

  OpDefBuilder("Cumsum", "CumsumTest")
    .Input("InputNCHW")
    .Output("OutputNCHW")
    .AddIntArg("axis", axis)
    .AddIntArg("exclusive", exclusive)
    .AddIntArg("reverse", reverse)
    .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
    .AddIntArg("has_data_format", 1)
    .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::CPU);

  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);

  net.AddInputFromArray<CPU, T>("ExpectedOutput", shape, output);
  ExpectTensorNear<T>(*net.GetOutput("ExpectedOutput"),
                      *net.GetOutput("Output"));
}
}  // namespace

TEST_F(CumsumOpTest, HasDataFormatCPU) {
  SimpleTestWithDataFormat<float>(
      {2, 2, 2, 2},
      {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.},
      0, 0, 0,
      {0., 1., 2., 3., 4., 5., 6., 7., 8., 10., 12., 14., 16., 18., 20., 22.});
  SimpleTestWithDataFormat<float>(
      {2, 2, 2, 2},
      {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.},
      1, 0, 0,
      {0., 1., 2., 3., 4., 6., 8., 10., 8., 9., 10., 11., 20., 22., 24., 26.});
  SimpleTestWithDataFormat<float>(
      {2, 2, 2, 2},
      {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.},
      0, 1, 0,
      {0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 4., 5., 6., 7.});
  SimpleTestWithDataFormat<float>(
      {2, 2, 2, 2},
      {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.},
      0, 0, 1,
      {8., 10., 12., 14., 16., 18., 20., 22., 8., 9., 10., 11., 12., 13., 14.,
      15.});
  SimpleTestWithDataFormat<float>(
      {2, 2, 2, 2},
      {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.},
      1, 1, 1,
      {4., 5., 6., 7., 0., 0., 0., 0., 12., 13., 14., 15., 0., 0., 0., 0.});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
