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
#include "mace/kernels/eltwise.h"

namespace mace {
namespace ops {
namespace test {

class ScalarMathOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T, typename DstType>
void ScalarMathTest(const kernels::EltwiseType type,
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
      .OutputType({kernels::IsLogicalType(type) ? DT_INT32 : DT_FLOAT})
      .Output("Output")
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp(D);


  auto expected = CreateTensor<DstType>({}, {output});

  ExpectTensorNear<DstType>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(ScalarMathOpTest, SimpleCPU) {
ScalarMathTest<DeviceType::CPU, float, float>(
    kernels::EltwiseType::SUM, 1, 2, 3, 3);
ScalarMathTest<DeviceType::CPU, float, float>(
    kernels::EltwiseType::SUB, 1, 2, 3, -1);
ScalarMathTest<DeviceType::CPU, float, float>(
    kernels::EltwiseType::PROD, 3, -2, 3, -6);
ScalarMathTest<DeviceType::CPU, float, float>(
    kernels::EltwiseType::DIV, 3, -2, 1, -1.5);
ScalarMathTest<DeviceType::CPU, float, float>(
    kernels::EltwiseType::MIN, 3, -2, 1, -2);
ScalarMathTest<DeviceType::CPU, float, float>(
    kernels::EltwiseType::MAX, 3, -2, 1, 3);
ScalarMathTest<DeviceType::CPU, float, float>(
    kernels::EltwiseType::NEG, 3, -2, 1, -3);
ScalarMathTest<DeviceType::CPU, float, float>(
    kernels::EltwiseType::ABS, 3, -2, 1, 3);
ScalarMathTest<DeviceType::CPU, float, float>(
    kernels::EltwiseType::SQR_DIFF, 3, -2, 1, 25);
ScalarMathTest<DeviceType::CPU, float, float>(
    kernels::EltwiseType::POW, 3, 1, 1, 3);
ScalarMathTest<DeviceType::CPU, float, int32_t>(
    kernels::EltwiseType::EQUAL, 3, 3, 1, 1);
}

TEST_F(ScalarMathOpTest, SimpleGPU) {
ScalarMathTest<DeviceType::GPU, float, float>(
    kernels::EltwiseType::SUM, 1, 2, 1, 3);
ScalarMathTest<DeviceType::GPU, float, float>(
    kernels::EltwiseType::SUB, 1, 2, 1, -1);
ScalarMathTest<DeviceType::GPU, float, float>(
    kernels::EltwiseType::PROD, 3, -2, 1, -6);
ScalarMathTest<DeviceType::GPU, float, float>(
    kernels::EltwiseType::DIV, 3, -2, 1, -1.5);
ScalarMathTest<DeviceType::GPU, float, float>(
    kernels::EltwiseType::MIN, 3, -2, 1, -2);
ScalarMathTest<DeviceType::GPU, float, float>(
    kernels::EltwiseType::MAX, 3, -2, 1, 3);
ScalarMathTest<DeviceType::GPU, float, float>(
    kernels::EltwiseType::NEG, 3, -2, 1, -3);
ScalarMathTest<DeviceType::GPU, float, float>(
    kernels::EltwiseType::ABS, 3, -2, 1, 3);
ScalarMathTest<DeviceType::GPU, float, float>(
    kernels::EltwiseType::SQR_DIFF, 3, -2, 1, 25);
ScalarMathTest<DeviceType::GPU, float, float>(
    kernels::EltwiseType::POW, 3, 1, 1, 3);
ScalarMathTest<DeviceType::GPU, float, int32_t>(
    kernels::EltwiseType::EQUAL, 3, 3, 1, 1);
}
}  // namespace test
}  // namespace ops
}  // namespace mace
