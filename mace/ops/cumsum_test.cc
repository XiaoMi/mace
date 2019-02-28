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

#include <functional>
#include <vector>

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class CumsumOpTest : public OpsTestBase {};

namespace {
void SimpleTest() {
  // Construct graph
  OpsTestNet net;

  net.AddInputFromArray<DeviceType::CPU, float>("Input", {2, 2, 2, 2},
      {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});

  OpDefBuilder("Cumsum", "CumsumTest")
    .Input("Input")
    .Output("Output")
    .AddIntArg("axis", 1)
    .AddIntArg("exclusive", 1)
    .AddIntArg("reverse", 1)
    .AddIntArg("T", static_cast<int>(DataTypeToEnum<float>::value))
    .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::CPU);

  auto expected = net.CreateTensor<float>({2, 2, 2, 2},
      {4., 5., 6., 7., 0., 0., 0., 0., 12., 13., 14., 15., 0., 0., 0., 0.});
  ExpectTensorNear<float, float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(CumsumOpTest, CPU) {
  SimpleTest();
}

}  // namespace test
}  // namespace ops
}  // namespace mace
