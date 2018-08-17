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

class ReshapeTest : public OpsTestBase {};

namespace {
void TestReshape(const std::vector<index_t> &org_shape,
                 const std::vector<int> &output_shape,
                 const std::vector<index_t> &res_shape) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Reshape", "ReshapeTest")
      .Input("Input")
      .Input("Shape")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::CPU, float>("Input", org_shape);
  net.AddInputFromArray<DeviceType::CPU, int32_t>(
      "Shape",
      {static_cast<index_t>(output_shape.size())},
      output_shape);

  // Run
  net.RunOp();

  auto input = net.GetTensor("Input");
  auto output = net.GetTensor("Output");

  EXPECT_THAT(output->shape(), ::testing::ContainerEq(res_shape));

  const float *input_ptr = input->data<float>();
  const float *output_ptr = output->data<float>();
  const int size = output->size();
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(input_ptr[i], output_ptr[i]);
  }
}
}  // namespace

TEST_F(ReshapeTest, Simple) {
  TestReshape({1, 2, 3, 4}, {1, 2, -1, 4}, {1, 2, 3, 4});
  TestReshape({1, 2, 3, 4}, {1, 2, -1, 2}, {1, 2, 6, 2});
  TestReshape({1, 2, 3, 4}, {1, -1, 3, 2}, {1, 4, 3, 2});
  TestReshape({1, 2, 3, 4}, {2, 2, 3, 2}, {2, 2, 3, 2});
}

TEST_F(ReshapeTest, Complex) {
  TestReshape({1, 2, 3, 4}, {-1}, {24});
  TestReshape({1, 2, 3, 4}, {1, -1}, {1, 24});
  TestReshape({1, 2, 3, 4}, {-1, 1}, {24, 1});
  TestReshape({1, 2, 3, 4}, {1, 3, 8}, {1, 3, 8});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
