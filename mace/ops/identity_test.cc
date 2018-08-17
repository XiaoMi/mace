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

#include "gmock/gmock.h"
#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class IdentityOpTest : public OpsTestBase {};

namespace {
void TestIdentity(const std::vector<index_t> &shape) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Identity", "IdentityTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::CPU, float>("Input", shape);

  // Run
  net.RunOp();

  auto input = net.GetTensor("Input");
  auto output = net.GetTensor("Output");

  EXPECT_THAT(output->shape(), ::testing::ContainerEq(shape));

  const float *input_ptr = input->data<float>();
  const float *output_ptr = output->data<float>();
  const int size = output->size();
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(input_ptr[i], output_ptr[i]);
  }
}
}  // namespace

TEST_F(IdentityOpTest, TestIdentity) {
  TestIdentity({1, 2, 3, 4});
  TestIdentity({1});
  TestIdentity({});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
