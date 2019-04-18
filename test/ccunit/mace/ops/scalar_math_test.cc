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
#include "mace/ops/eltwise.h"

namespace mace {
namespace ops {
namespace test {

class ScalarMathOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T, typename DstType>
void ScalarMathTest(const ops::EltwiseType type,
                    const T input0,
                    const T input1,
                    const float x,
                    const DstType output) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, T>("Input0", {}, {input0});
  net.AddInputFromArray<D, T>("Input1", {}, {input1});

  OpDefBuilder("ScalarMath", "ScalarMathTest")
      .Input("Input0")
      .Input("Input1")
      .AddIntArg("T", DataTypeToEnum<T>::v())
      .AddIntArg("type", static_cast<int>(type))
      .AddFloatArg("scalar_input", x)
      .OutputType({ops::IsLogicalType(type) ? DT_INT32 : DT_FLOAT})
      .Output("Output")
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp(D);


  auto expected = net.CreateTensor<DstType>({}, {output});

  ExpectTensorNear<DstType>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(ScalarMathOpTest, SimpleCPU) {
  ScalarMathTest<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUM, 1, 2, 3, 3);
  ScalarMathTest<DeviceType::CPU, float, float>(
      ops::EltwiseType::SUB, 1, 2, 3, -1);
  ScalarMathTest<DeviceType::CPU, float, float>(
      ops::EltwiseType::PROD, 3, -2, 3, -6);
  ScalarMathTest<DeviceType::CPU, float, float>(
      ops::EltwiseType::DIV, 3, -2, 1, -1.5);
  ScalarMathTest<DeviceType::CPU, float, float>(
      ops::EltwiseType::FLOOR_DIV, 3, -2, 1, -2);
  ScalarMathTest<DeviceType::CPU, float, float>(
      ops::EltwiseType::FLOOR_DIV, 3, 2, 1, 1);
  ScalarMathTest<DeviceType::CPU, float, float>(
      ops::EltwiseType::MIN, 3, -2, 1, -2);
  ScalarMathTest<DeviceType::CPU, float, float>(
      ops::EltwiseType::MAX, 3, -2, 1, 3);
  ScalarMathTest<DeviceType::CPU, float, float>(
      ops::EltwiseType::NEG, 3, -2, 1, -3);
  ScalarMathTest<DeviceType::CPU, float, float>(
      ops::EltwiseType::ABS, 3, -2, 1, 3);
  ScalarMathTest<DeviceType::CPU, float, float>(
      ops::EltwiseType::SQR_DIFF, 3, -2, 1, 25);
  ScalarMathTest<DeviceType::CPU, float, float>(
      ops::EltwiseType::POW, 3, 1, 1, 3);
  ScalarMathTest<DeviceType::CPU, float, int32_t>(
      ops::EltwiseType::EQUAL, 3, 3, 1, 1);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
