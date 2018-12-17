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

class TargetRMSNormOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestTargetRMSNorm(const std::vector<index_t> &input_shape,
                       const std::vector<T> &input,
                       const float target_rms,
                       const std::vector<T> &output) {
  OpsTestNet net;
  net.AddInputFromArray<CPU, T>(MakeString("Input"),
                                input_shape,
                                input);

  OpDefBuilder("TargetRMSNorm", "TargetRMSNormTest")
      .Input("Input")
      .AddFloatArg("target_rms", target_rms)
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  net.RunOp();

  net.AddInputFromArray<CPU, T>("ExpectedOutput", input_shape, output);
  ExpectTensorNear<T>(*net.GetOutput("ExpectedOutput"),
                      *net.GetOutput("Output"));
}
}  // namespace

TEST_F(TargetRMSNormOpTest, SimpleTest) {
  TestTargetRMSNorm<DeviceType::CPU, float>(
    {1, 3, 3},
    {1, 2, 3,
     2, 3, 4,
     3, 4, 5},
     1.0,
    {0.46291, 0.92582, 1.38873,
     0.64327, 0.9649, 1.28654,
     0.734847, 0.979796, 1.224745});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
