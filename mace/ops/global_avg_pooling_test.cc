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

class GlobalAvgPoolingOpTest : public OpsTestBase {};

TEST_F(GlobalAvgPoolingOpTest, 3x7x7_CPU) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("GlobalAvgPooling", "GlobalAvgPoolingTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  std::vector<float> input(147);
  for (int i = 0; i < 147; ++i) {
    input[i] = i / 49 + 1;
  }
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 3, 7, 7}, input);

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 3, 1, 1}, {1, 2, 3});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
