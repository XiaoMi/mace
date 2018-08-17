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

class FillTest : public OpsTestBase {};

namespace {
void TestFill(const std::vector<int32_t> &shape,
              const float &value) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Fill", "FillTest")
      .Input("Shape")
      .Input("Value")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, int32_t>(
      "Shape",
      {static_cast<index_t>(shape.size())},
      shape);

  net.AddInputFromArray<DeviceType::CPU, float>("Value", {}, {value});

  // Run
  net.RunOp();

  auto output = net.GetTensor("Output");

  for (index_t i = 0; i < output->dim_size(); ++i) {
    EXPECT_EQ(output->dim(i), shape[i]);
  }

  const float *output_ptr = output->data<float>();
  const index_t size = output->size();
  for (index_t i = 0; i < size; ++i) {
    EXPECT_EQ(output_ptr[i], value);
  }
}
}  // namespace

TEST_F(FillTest, Simple) {
  TestFill({3, 2, 1}, 5.0f);
  TestFill({1, 3}, -1.0f);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
